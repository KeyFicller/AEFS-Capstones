"""Supervised fine-tuning on the assistant-only loss mask.

`collate_sft` shifts and masks the batch, and the model loss already ignores
-100, so the mask is never multiplied in a second time. `max_steps` wins over
`epochs` when set; batches cycle through the train split, so a split smaller
than one batch still fills every step.
"""

import logging
import math
import time
from pathlib import Path

import torch
from torch import nn

from tiny_llm_pipeline.config import DEFAULT_TRAIN, TrainConfig, lr_at
from tiny_llm_pipeline.data import SFTBundle, collate_sft
from tiny_llm_pipeline.model import load_ckpt, save_ckpt
from tiny_llm_pipeline.tokenizer import Tok
from tiny_llm_pipeline.train.pretrain import _append_log, _device, _optimizer

logger = logging.getLogger(__name__)


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
) -> Path:
    """Fine-tune from `base_ckpt` (or `resume`) and return the checkpoint path.

    The start point is `base_ckpt` unless `resume` is given. The step counter
    starts at 0 for a fresh stage: `base_ckpt` is pretraining's checkpoint, so
    its `step` belongs to a different run and carrying it over would make
    `max_steps` absolute and silently train fewer steps than asked. Optimizer
    and schedule match pretraining.
    """
    rows = sft_bundle.train
    if not rows:
        raise ValueError("sft train split is empty")
    device = _device()
    start = base_ckpt if resume is None else resume
    loaded = load_ckpt(start, expect_tokenizer_hash=tok_hash)
    model = loaded["model"].to(device)
    opt = _optimizer(model, train_cfg)
    step = int(loaded["step"]) if resume is not None else 0
    payload: dict[str, object] | None = None
    if resume is not None:
        payload = torch.load(Path(resume), map_location="cpu", weights_only=False)
        state = payload.get("optimizer")
        if isinstance(state, dict):
            opt.load_state_dict(state)
    tokens = int(payload["tokens"]) if payload and "tokens" in payload else 0

    batch_size = train_cfg.batch_size
    per_epoch = max(1, math.ceil(len(rows) / batch_size))
    total = max_steps if max_steps is not None else max(1, epochs) * per_epoch
    horizon = max(1, total)

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ckpt_path = dest / "ckpt.pt"
    log_path = dest / "train_log.jsonl"

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
    model.train()
    while step < total:
        step += 1
        first = ((step - 1) * batch_size) % len(rows)
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
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()
        tokens += int(batch["loss_mask"].sum())
        record: dict[str, float | int] = {
            "step": step,
            "loss": float(loss.detach()),
            "lr": lr,
            "tokens": tokens,
            "elapsed": round(time.perf_counter() - started, 3),
        }
        _append_log(log_path, record)
        logger.info("step=%d loss=%.4f lr=%.6g tokens=%d", step, record["loss"], lr, tokens)
    save_ckpt(
        ckpt_path, model, model.cfg, step, tok_hash, optimizer=opt, seed=seed, tokens=tokens
    )
    return ckpt_path


def eval_sft_ppl(
    ckpt: Path | str, holdout: list[dict], tok: Tok, device: str | torch.device
) -> float:
    """Response-token perplexity over `holdout`, same template and mask as training.

    Weight each row by its non-masked token count. An empty holdout returns inf.
    """
    if not holdout:
        return math.inf
    loaded = load_ckpt(ckpt)
    model = loaded["model"].to(device)
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
