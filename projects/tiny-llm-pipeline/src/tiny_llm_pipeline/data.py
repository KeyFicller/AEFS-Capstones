"""Dataset download, document splits, packed token bins, and batch collation.

Pretrain bins store a token stream. `max_seq_len` is the `PackedDataset` window,
not a minimum length for writing the file. SFT and DPO prompts are cut from
disjoint user texts.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import Tensor

from tiny_llm_pipeline.config import DEFAULT_MODEL
from tiny_llm_pipeline.tokenizer import Tok, encode_chat

logger = logging.getLogger(__name__)

_EOS = "<|eos|>"
_PAD_ID = 0
_IGNORE_INDEX = -100
_CHAT_MARKERS = re.compile(r"<\|im_start\|>|<\|im_end\|>")


class ConversationTurn(TypedDict):
    """One chat turn in an SFT row."""

    role: str
    content: str


class SFTExample(TypedDict):
    """One SFT jsonl row."""

    conversations: list[ConversationTurn]


class PreferenceExample(TypedDict):
    """One preference row. Extra keys may be present at runtime."""

    prompt: str
    chosen: str
    rejected: str


class PretrainBatch(TypedDict):
    """Stacked pretrain windows. The training loop shifts for next-token loss."""

    input_ids: Tensor


class SFTBatch(TypedDict):
    """Shifted SFT batch. Non-assistant labels are -100."""

    input_ids: Tensor
    labels: Tensor
    loss_mask: Tensor


class DPOBatch(TypedDict):
    """Unshifted preference batch. `seq_logprob` shifts later."""

    chosen_input_ids: Tensor
    chosen_mask: Tensor
    rejected_input_ids: Tensor
    rejected_mask: Tensor


@dataclass(frozen=True)
class SFTBundle:
    """File-order SFT split. The three user-text sets do not overlap."""

    train: list[SFTExample]
    holdout: list[SFTExample]
    dpo_prompts: list[str]


def fetch_minimind(
    out_dir: Path | str,
    filename: str,
    repo: str = "jingyaogong/minimind_dataset",
) -> Path:
    """Download one file with `huggingface_hub.hf_hub_download`.

    On any failure raise `FileNotFoundError` naming `repo` and `filename`.
    Do not switch mirrors or datasets.
    """
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(repo_id=repo, filename=filename, local_dir=dest)
    except Exception as exc:
        raise FileNotFoundError(f"failed to download {filename} from {repo}") from exc
    return Path(downloaded)


def prepare_pretrain(
    raw_jsonl: Path | str,
    out_dir: Path | str,
    tok: Tok,
    max_tokens: int,
    split_ratio: float = 0.995,
) -> tuple[Path, Path]:
    """Write `train.bin` and `val.bin` of uint16 token ids.

    Split each line on `<|im_start|>` / `<|im_end|>` and drop those markers.
    Drop passages shorter than 50 characters, then sha1-dedupe. Append
    `<|eos|>` to each passage. Cap the token stream at `max_tokens` before
    the document-level split. A single passage goes entirely to train and
    leaves `val.bin` empty. Keep a short tail; do not drop a passage for
    being shorter than 512.
    """
    eos = tok.encode(_EOS)[0]
    kept: list[np.ndarray] = []
    seen: set[str] = set()
    used = 0
    # Stream so a multi-gigabyte jsonl is not loaded at once. Stop once the
    # token cap is hit; later documents would be dropped by the cap anyway.
    with Path(raw_jsonl).open(encoding="utf-8") as source:
        for line in source:
            if used >= max_tokens:
                break
            if not line.strip():
                continue
            raw = str(json.loads(line).get("text", ""))
            for text in _passages(raw):
                if used >= max_tokens:
                    break
                if len(text) < 50:
                    continue
                digest = hashlib.sha1(text.encode()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                ids = tok.encode(text) + [eos]
                chunk = np.asarray(ids[: max_tokens - used], dtype=np.uint16)
                kept.append(chunk)
                used += int(chunk.shape[0])
                if len(kept) % 20_000 == 0:
                    logger.info("pretrain tokens=%d docs=%d", used, len(kept))

    n_val = int(len(kept) * (1 - split_ratio)) if len(kept) > 1 else 0
    val_docs, train_docs = kept[:n_val], kept[n_val:]
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    return _dump_bin(train_docs, dest / "train.bin"), _dump_bin(val_docs, dest / "val.bin")


def _passages(text: str) -> list[str]:
    """Split a pretrain line into plain passages, dropping chat-shell markers."""
    return [part.strip() for part in _CHAT_MARKERS.split(text) if part.strip()]


def prepare_sft(
    raw_jsonl: Path | str,
    tok: Tok,
    holdout_n: int = 200,
    dpo_prompt_n: int = 1000,
) -> SFTBundle:
    """Split SFT jsonl in file order, without shuffling.

    Drop rows whose `encode_chat` length exceeds `ModelConfig.max_seq_len`.
    Then `holdout` is the first `holdout_n` rows, `dpo_prompts` is the first
    user turn of the next `dpo_prompt_n` rows, and `train` is the rest.
    """
    rows = [
        row
        for row in _load_sft(Path(raw_jsonl))
        if len(encode_chat(list(row["conversations"]), tok)[0]) <= DEFAULT_MODEL.max_seq_len
    ]
    holdout = rows[:holdout_n]
    rest = rows[holdout_n:]
    prompts = [_first_user(row) for row in rest[:dpo_prompt_n]]
    return SFTBundle(train=rest[dpo_prompt_n:], holdout=holdout, dpo_prompts=prompts)


class PackedDataset:
    """Cycling windows over a uint16 memmap.

    Window length is `min(seq_len, max(2, len(data)))`. `__getitem__` wraps
    the index so a short bin still yields a window.
    """

    def __init__(self, bin_path: Path | str, seq_len: int) -> None:
        path = Path(bin_path)
        # A one-document split leaves a 0-byte val.bin. mmap rejects that file.
        if path.stat().st_size == 0:
            self._data = np.zeros(0, dtype=np.uint16)
        else:
            self._data = np.memmap(path, dtype=np.uint16, mode="r")
        n = int(self._data.shape[0])
        self.seq = min(seq_len, max(2, n)) if n else 0
        self._windows = max(1, n // self.seq) if n and self.seq else 0

    def __len__(self) -> int:
        return self._windows

    def __getitem__(self, index: int) -> Tensor:
        if self._windows == 0:
            raise IndexError(index)
        start = (index % self._windows) * self.seq
        chunk = np.asarray(self._data[start : start + self.seq], dtype=np.int64)
        return torch.tensor(chunk)


def collate_pretrain(windows: list[Tensor]) -> PretrainBatch:
    """Stack `[T]` windows into `input_ids` of shape `[B, T]`. Do not shift."""
    return {"input_ids": torch.stack(windows)}


def collate_sft(rows: list[SFTExample], tok: Tok, max_seq_len: int) -> SFTBatch:
    """Shift each chat so tokens predict the next id, then right-pad.

    Truncate to `max_seq_len` before the shift. Pad id is 0. Padded and
    non-assistant positions have label -100 and `loss_mask` False.
    `loss_mask` matches `labels != -100`.
    """
    batch_ids: list[list[int]] = []
    batch_labels: list[list[int]] = []
    batch_mask: list[list[bool]] = []
    for row in rows:
        ids, mask = encode_chat(list(row["conversations"]), tok)
        ids, mask = ids[:max_seq_len], mask[:max_seq_len]
        labels = [token if keep else _IGNORE_INDEX for token, keep in zip(ids[1:], mask[1:], strict=True)]
        batch_ids.append(ids[:-1])
        batch_labels.append(labels)
        batch_mask.append(mask[1:])
    return {
        "input_ids": _right_pad(batch_ids, _PAD_ID),
        "labels": _right_pad(batch_labels, _IGNORE_INDEX),
        "loss_mask": _right_pad_mask(batch_mask),
    }


def collate_dpo(rows: list[PreferenceExample], tok: Tok, max_seq_len: int) -> DPOBatch:
    """Encode chosen and rejected chats without shifting, then right-pad.

    Each side is a user turn plus the assistant reply. Pad id is 0 and the
    pad mask is False.
    """
    chosen_ids: list[list[int]] = []
    chosen_mask: list[list[bool]] = []
    rejected_ids: list[list[int]] = []
    rejected_mask: list[list[bool]] = []
    for row in rows:
        chosen = _preference_side(row["prompt"], row["chosen"], tok, max_seq_len)
        rejected = _preference_side(row["prompt"], row["rejected"], tok, max_seq_len)
        chosen_ids.append(chosen[0])
        chosen_mask.append(chosen[1])
        rejected_ids.append(rejected[0])
        rejected_mask.append(rejected[1])
    return {
        "chosen_input_ids": _right_pad(chosen_ids, _PAD_ID),
        "chosen_mask": _right_pad_mask(chosen_mask),
        "rejected_input_ids": _right_pad(rejected_ids, _PAD_ID),
        "rejected_mask": _right_pad_mask(rejected_mask),
    }


def _dump_bin(docs: list[np.ndarray], path: Path) -> Path:
    if docs:
        np.concatenate(docs).tofile(path)
    else:
        np.asarray([], dtype=np.uint16).tofile(path)
    return path


def _load_sft(path: Path) -> list[SFTExample]:
    rows: list[SFTExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _first_user(row: SFTExample) -> str:
    for message in row["conversations"]:
        if message["role"] == "user":
            return message["content"]
    raise ValueError("SFT row has no user turn")


def _preference_side(
    prompt: str, reply: str, tok: Tok, max_seq_len: int
) -> tuple[list[int], list[bool]]:
    ids, mask = encode_chat(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}],
        tok,
    )
    return ids[:max_seq_len], mask[:max_seq_len]


def _right_pad(rows: list[list[int]], pad: int) -> Tensor:
    width = max((len(row) for row in rows), default=0)
    out = torch.full((len(rows), width), pad, dtype=torch.long)
    for i, row in enumerate(rows):
        if row:
            out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def _right_pad_mask(rows: list[list[bool]]) -> Tensor:
    width = max((len(row) for row in rows), default=0)
    out = torch.zeros((len(rows), width), dtype=torch.bool)
    for i, row in enumerate(rows):
        if row:
            out[i, : len(row)] = torch.tensor(row, dtype=torch.bool)
    return out
