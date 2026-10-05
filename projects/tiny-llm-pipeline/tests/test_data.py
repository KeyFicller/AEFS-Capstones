import json
from pathlib import Path

import numpy as np

from tiny_llm_pipeline.data import (
    DATASET_REVISION,
    collate_sft,
    fetch_minimind,
    prepare_pretrain,
    prepare_sft,
    write_sft_splits,
    write_tokenizer_sample,
)


def test_uint16_roundtrip(tmp_path, tok, tiny_pretrain_jsonl) -> None:
    train_bin, _val_bin = prepare_pretrain(tiny_pretrain_jsonl, tmp_path, tok, max_tokens=4096)
    arr = np.memmap(train_bin, dtype=np.uint16, mode="r")
    assert arr.dtype == np.uint16 and arr.max() < tok.vocab_size


def test_pretrain_strips_chat_markers(tmp_path, tok) -> None:
    body = "今天天气真好，我们出去玩吧。" * 4
    other = "明天一起去公园里慢慢散步。" * 4
    raw = f"<|im_start|>{body}<|im_end|> <|im_start|>短<|im_end|> <|im_start|>{other}<|im_end|>"
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"text": raw}, ensure_ascii=False) + "\n", encoding="utf-8")
    train_bin, _val_bin = prepare_pretrain(source, tmp_path, tok, max_tokens=4096)
    arr = np.memmap(train_bin, dtype=np.uint16, mode="r")
    decoded = tok.decode(arr.tolist())
    assert "<|im_start|>" not in decoded and "<|im_end|>" not in decoded
    assert decoded == body + other
    assert arr.tolist().count(tok.encode("<|eos|>")[0]) == 2


def test_three_splits_disjoint(tiny_sft_jsonl, tok) -> None:
    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=2, dpo_prompt_n=3)
    train_qs = {
        message["content"].strip()
        for sample in bundle.train
        for message in sample["conversations"]
        if message["role"] == "user"
    }
    hold_qs = {
        message["content"].strip()
        for sample in bundle.holdout
        for message in sample["conversations"]
        if message["role"] == "user"
    }
    dpo_qs = {prompt.strip() for prompt in bundle.dpo_prompts}
    assert not (train_qs & hold_qs) and not (train_qs & dpo_qs) and not (hold_qs & dpo_qs)


def test_sft_collate_shapes_and_mask(tiny_sft_jsonl, tok) -> None:
    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=1, dpo_prompt_n=1)
    batch = collate_sft(bundle.train[:4], tok, max_seq_len=128)
    assert batch["input_ids"].shape == batch["loss_mask"].shape
    assert batch["loss_mask"].sum() > 0
    assert (batch["labels"][~batch["loss_mask"]] == -100).all()


def test_fetch_minimind_asks_for_the_dataset_repo(tmp_path, monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_hf_hub_download(**kwargs):
        calls.update(kwargs)
        dest = Path(kwargs["local_dir"]) / str(kwargs["filename"])
        dest.write_text("{}", encoding="utf-8")
        return str(dest)

    monkeypatch.setattr("tiny_llm_pipeline.data.hf_hub_download", fake_hf_hub_download)
    fetched = fetch_minimind(tmp_path, "sft_mini_512.jsonl")

    # `jingyaogong/minimind_dataset` is a dataset repo whose `main` no longer
    # holds these files: the default repo_type ("model") answers 401, and an
    # unpinned revision 404s.
    assert calls["repo_type"] == "dataset"
    assert calls["revision"] == DATASET_REVISION
    assert calls["repo_id"] == "jingyaogong/minimind_dataset"
    assert fetched == tmp_path / "sft_mini_512.jsonl"


def test_tokenizer_sample_dedupes_and_drops_short(tmp_path, tiny_pretrain_jsonl) -> None:
    out = write_tokenizer_sample(tiny_pretrain_jsonl, tmp_path / "sample.txt", max_bytes=1_000_000)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert len(lines[0]) >= 50


def test_tokenizer_sample_caps_bytes_at_a_character_boundary(tmp_path) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"text": "今天天气真好，我们出去玩吧。" * 4}, ensure_ascii=False) + "\n", encoding="utf-8")
    out = write_tokenizer_sample(source, tmp_path / "sample.txt", max_bytes=40)
    written = out.read_bytes()
    assert 0 < len(written) <= 40
    assert written.decode("utf-8")


def test_write_sft_splits_round_trips_every_split(tmp_path, tiny_sft_jsonl, tok) -> None:
    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=2, dpo_prompt_n=3)
    paths = write_sft_splits(bundle, tmp_path)
    assert set(paths) == {"train", "holdout", "dpo_prompts"}
    train_rows = [json.loads(line) for line in paths["train"].read_text(encoding="utf-8").splitlines()]
    holdout_rows = [
        json.loads(line) for line in paths["holdout"].read_text(encoding="utf-8").splitlines()
    ]
    prompts = [json.loads(line) for line in paths["dpo_prompts"].read_text(encoding="utf-8").splitlines()]
    assert train_rows == bundle.train
    assert holdout_rows == bundle.holdout
    assert [row["prompt"] for row in prompts] == bundle.dpo_prompts


def test_prepare_sft_drops_rows_that_repeat_another_split(tmp_path, tok) -> None:
    # The real corpus repeats user turns, so index-only slicing lets a train
    # row share its user text with the holdout or the DPO prompts. A repeat in
    # any turn counts, not just the first one.
    rows = [
        {
            "conversations": [
                {"role": "user", "content": f"问题{i}？"},
                {"role": "assistant", "content": f"回答{i}。"},
            ]
        }
        for i in range(12)
    ]
    rows[5]["conversations"][0]["content"] = "问题0？"
    rows[7]["conversations"] = [
        {"role": "user", "content": "问题7？"},
        {"role": "assistant", "content": "回答7。"},
        {"role": "user", "content": "问题1？"},
        {"role": "assistant", "content": "回答7b。"},
    ]
    raw = tmp_path / "sft.jsonl"
    raw.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    bundle = prepare_sft(raw, tok, holdout_n=2, dpo_prompt_n=3)

    train_qs = {m["content"].strip() for s in bundle.train for m in s["conversations"] if m["role"] == "user"}
    hold_qs = {m["content"].strip() for s in bundle.holdout for m in s["conversations"] if m["role"] == "user"}
    dpo_qs = {prompt.strip() for prompt in bundle.dpo_prompts}
    assert not (train_qs & hold_qs) and not (train_qs & dpo_qs) and not (hold_qs & dpo_qs)
    assert len(bundle.dpo_prompts) == 3


def test_prepare_sft_keeps_a_record_spanning_raw_newlines(tmp_path, tok) -> None:
    # `sft_mini_512.jsonl` holds unescaped newlines inside `content`, so the
    # record is split across physical lines but is still one conversation.
    raw = tmp_path / "sft.jsonl"
    body = "第一行\n第二行\n第三行"
    rows = [
        {"conversations": [{"role": "user", "content": f"问题{i}"}, {"role": "assistant", "content": body}]}
        for i in range(4)
    ]
    lines = [json.dumps(row, ensure_ascii=False).replace("\\n", "\n") for row in rows]
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bundle = prepare_sft(raw, tok, holdout_n=1, dpo_prompt_n=1)
    assert len(bundle.train) + len(bundle.holdout) + len(bundle.dpo_prompts) == len(rows)
    assert bundle.holdout[0]["conversations"][-1]["content"] == body
