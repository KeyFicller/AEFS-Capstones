"""Supervised fine-tuning on the assistant-only loss mask.

`collate_sft` shifts and masks the batch, and the model loss already ignores
-100, so the mask is never multiplied in a second time. `max_steps` wins over
`epochs` when set; batches cycle through the train split, so a split smaller
than one batch still fills every step.

Checkpoints, the monitor plot, and held-out eval follow pretraining: save
every `ckpt_every` steps (and on Ctrl-C), write `monitor.png` every
`plot_every` steps, and score the holdout split at each save. `--resume`
continues that step's data cursor.
"""

import json
import logging
import math
import random
import signal
import time
from pathlib import Path

import torch
from torch import nn

from tiny_llm_pipeline.config import DEFAULT_TRAIN, TrainConfig, lr_at
from tiny_llm_pipeline.data import SFTBundle, collate_sft
from tiny_llm_pipeline.lora import LoRAConfig, inject_lora
from tiny_llm_pipeline.model import TinyLM, load_ckpt, load_lora_ckpt, save_ckpt, save_lora_ckpt
from tiny_llm_pipeline.tokenizer import Tok, encode_chat
from tiny_llm_pipeline.train import pretrain as _pretrain
from tiny_llm_pipeline.train.pretrain import (
    _append_log,
    _device,
    _interrupt_requested,
    _on_sigint,
    _optimizer,
    _truncate_log,
)

logger = logging.getLogger(__name__)

_VAL_SAMPLES = 5
_VAL_NEW = 32
_END = "<|end|>"


def train_sft(
    sft_bundle: SFTBundle,
    out_dir: Path | str,
    *,
    tok: Tok,
    base_ckpt: Path | str,
    tok_hash: str,
    epochs: int = 2,
    max_steps: int | None = None,
    seed: int = 42,
    resume: Path | str | None = None,
    train_cfg: TrainConfig = DEFAULT_TRAIN,
    ckpt_every: int = 500,
    plot_every: int = 10,
    val_every: int = 1000,
    lora: LoRAConfig | None = None,
) -> Path:
    """Fine-tune from `base_ckpt` (or `resume`) and return the checkpoint path.

    The start point is `base_ckpt` unless `resume` is given. The step counter
    starts at 0 for a fresh stage: `base_ckpt` is pretraining's checkpoint, so
    its `step` belongs to a different run and carrying it over would make
    `max_steps` absolute and silently train fewer steps than asked. Optimizer
    and schedule match pretraining.

    `lora` injects adapters into the attention projections and writes adapter
    checkpoints instead of full weights; `rank <= 0` behaves like `None`.

    `resume` continues the saved step, optimizer, token count, and row cursor.
    The first Ctrl-C finishes the current step, writes a checkpoint, and returns.
    Each save records holdout response-token perplexity when the split is
    non-empty. The lowest one so far is also written to `best.pt`, and the
    return value is `best.pt` when it exists so a later step cannot hide it.
    `val_every` prints greedy answers for 5 rows drawn for that step.
    """
    if val_every == 0 or val_every < -1:
        raise ValueError("val_every must be -1 or a positive step interval")
    rows = sft_bundle.train
    if not rows:
        raise ValueError("sft train split is empty")
    device = _device()
    start = base_ckpt if resume is None else resume
    lora_cfg = lora if (lora is not None and lora.enabled) else None
    if resume is not None and lora_cfg is not None:
        model, _ = load_lora_ckpt(resume, expect_tokenizer_hash=tok_hash)
        step = int(torch.load(Path(resume), map_location="cpu", weights_only=False)["step"])
    else:
        loaded = load_ckpt(start, expect_tokenizer_hash=tok_hash)
        model = loaded["model"]
        step = int(loaded["step"]) if resume is not None else 0
        if lora_cfg is not None:
            inject_lora(model, lora_cfg)
    model = model.to(device)
    opt = _optimizer(model, train_cfg)
    payload: dict[str, object] | None = None
    saved_tokens: int | None = None
    saved_cursor: int | None = None
    if resume is not None:
        payload = torch.load(Path(resume), map_location="cpu", weights_only=False)
        state = payload.get("optimizer")
        if not isinstance(state, dict):
            raise RuntimeError(f"checkpoint has no optimizer state: {resume}")
        opt.load_state_dict(state)
        if "tokens" in payload:
            saved_tokens = int(payload["tokens"])
        if "cursor" in payload:
            saved_cursor = int(payload["cursor"])

    batch_size = train_cfg.batch_size
    per_epoch = max(1, math.ceil(len(rows) / batch_size))
    total = max_steps if max_steps is not None else max(1, epochs) * per_epoch
    horizon = max(1, total)
    tokens = 0 if saved_tokens is None else saved_tokens
    cursor = step * batch_size if saved_cursor is None else saved_cursor

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ckpt_path = dest / "ckpt.pt"
    best_path = dest / "best.pt"
    log_path = dest / "train_log.jsonl"
    elapsed_offset = _truncate_log(log_path, step) if resume is not None else 0.0
    best_ppl = _best_ppl(log_path)

    torch.manual_seed(seed)
    max_seq_len = model.cfg.max_seq_len
    started = time.perf_counter()
    logger.info(
        "sft device=%s batch=%d rows=%d horizon=%d step=%d",
        device.type,
        batch_size,
        len(rows),
        horizon,
        step,
    )

    def _save(to: Path = ckpt_path) -> None:
        if lora_cfg is not None:
            save_lora_ckpt(
                to,
                model,
                model.cfg,
                lora_cfg,
                base_ckpt,
                step,
                tok_hash,
                optimizer=opt,
                seed=seed,
                tokens=tokens,
                cursor=cursor,
            )
            return
        save_ckpt(
            to,
            model,
            model.cfg,
            step,
            tok_hash,
            optimizer=opt,
            seed=seed,
            tokens=tokens,
            cursor=cursor,
        )

    _pretrain._INTERRUPT = False
    previous_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        model.train()
        while step < total:
            step += 1
            first = cursor % len(rows)
            cursor += batch_size
            batch_rows = [rows[(first + offset) % len(rows)] for offset in range(batch_size)]
            batch = collate_sft(batch_rows, tok, max_seq_len)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lr = lr_at(step - 1, horizon, train_cfg)
            for group in opt.param_groups:
                group["lr"] = lr
            _, loss = model(input_ids, targets=labels)
            if loss is None:
                raise RuntimeError("sft forward returned no loss")
            if not torch.isfinite(loss):
                _save()
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            tokens += int(batch["loss_mask"].sum())
            wall = time.perf_counter() - started
            record: dict[str, float | int] = {
                "step": step,
                "loss": float(loss.detach()),
                "lr": lr,
                "tokens": tokens,
                "elapsed": round(elapsed_offset + wall, 3),
            }
            interrupted = _interrupt_requested()
            stop = step == total or interrupted
            if ckpt_every > 0 and (step % ckpt_every == 0 or stop):
                ppl = _response_ppl(model, sft_bundle.holdout, tok, device)
                if math.isfinite(ppl):
                    record["val_ppl"] = ppl
                _save()
                if ppl < best_ppl:
                    best_ppl = ppl
                    _save(best_path)
            _append_log(log_path, record)
            logger.info("step=%d loss=%.4f lr=%.6g tokens=%d", step, record["loss"], lr, tokens)
            if plot_every > 0 and (step % plot_every == 0 or stop):
                from tiny_llm_pipeline.viz import write_monitor

                write_monitor(log_path, dest / "monitor.png")
            if val_every > 0 and step % val_every == 0:
                _log_sft_samples(model, sft_bundle.holdout, tok, device, step, seed)
            if stop:
                if interrupted:
                    if ckpt_every <= 0:
                        _save()
                    logger.info("saved checkpoint at step %d after interrupt", step)
                elif not ckpt_path.is_file():
                    _save()
                return best_path if best_path.is_file() else ckpt_path
        return best_path if best_path.is_file() else ckpt_path
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        _pretrain._INTERRUPT = False


def _best_ppl(log_path: Path) -> float:
    """Lowest finite `val_ppl` already in the log. inf when there is none."""
    if not log_path.is_file():
        return math.inf
    best = math.inf
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line).get("val_ppl")
        if value is not None and math.isfinite(float(value)):
            best = min(best, float(value))
    return best


def eval_sft_ppl(
    ckpt: Path | str, holdout: list[dict], tok: Tok, device: str | torch.device
) -> float:
    """Response-token perplexity over `holdout`, same template and mask as training.

    Weight each row by its non-masked token count. An empty holdout returns inf.
    """
    loaded = load_ckpt(ckpt)
    return _response_ppl(loaded["model"].to(device), holdout, tok, device)


def _response_ppl(
    model: TinyLM, holdout: list[dict], tok: Tok, device: str | torch.device
) -> float:
    """Token-weighted response perplexity. An empty holdout returns inf."""
    if not holdout:
        return math.inf
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    try:
        with torch.no_grad():
            for row in holdout:
                batch = collate_sft([row], tok, model.cfg.max_seq_len)
                n = int(batch["loss_mask"].sum())
                if n == 0:
                    continue
                _, loss = model(
                    batch["input_ids"].to(device), targets=batch["labels"].to(device)
                )
                if loss is None:
                    return math.inf
                total += float(loss) * n
                count += n
    finally:
        model.train(was_training)
    if count == 0:
        return math.inf
    return math.exp(total / count)


def _sft_prompt(row: dict, tok: Tok) -> tuple[list[int], str]:
    """Ids up through `<|assistant|>`, and the gold text of the last reply."""
    messages = list(row["conversations"])
    gold = ""
    if messages and messages[-1]["role"] == "assistant":
        gold = str(messages[-1]["content"])
        messages = messages[:-1]
    ids, _mask = encode_chat(messages, tok, add_assistant=True)
    return ids, gold


def _holdout_sample(holdout: list[dict], step: int, seed: int) -> list[dict]:
    """Five rows for this step. The same step and seed repeat; a new step does not."""
    if len(holdout) <= _VAL_SAMPLES:
        return list(holdout)
    return random.Random(seed + step).sample(holdout, _VAL_SAMPLES)


def _log_sft_samples(
    model: TinyLM,
    holdout: list[dict],
    tok: Tok,
    device: torch.device,
    step: int,
    seed: int = 42,
) -> None:
    """Greedy replies for 5 holdout rows chosen for `step`, next to the gold answer."""
    rows = _holdout_sample(holdout, step, seed)
    if not rows:
        logger.info("step=%d sft samples skipped: empty holdout", step)
        return
    end_id = tok.encode(_END)[0]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for index, row in enumerate(rows):
                prompt_ids, gold = _sft_prompt(row, tok)
                prefix = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                pred = model.generate(
                    prefix,
                    _VAL_NEW,
                    temperature=0,
                    top_k=0,
                    top_p=1.0,
                    repetition_penalty=1.0,
                )
                new_ids = pred[0, len(prompt_ids) :].tolist()
                if end_id in new_ids:
                    new_ids = new_ids[: new_ids.index(end_id) + 1]
                logger.info("step=%d sft %d/%d", step, index + 1, len(rows))
                logger.info("prompt: %s", tok.decode(prompt_ids, skip_special_tokens=False))
                logger.info("gold: %s", gold)
                logger.info("pred: %s", tok.decode(new_ids, skip_special_tokens=False))
    finally:
        model.train(was_training)
