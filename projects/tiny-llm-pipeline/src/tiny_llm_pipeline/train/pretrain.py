"""Next-token pretraining on a packed uint16 bin.

The loop shifts internally: `x[:, :-1]` predicts `x[:, 1:]`. The token after
`<|eos|>` is ignored. An empty val bin makes `eval_ppl` return infinity.
"""

import json
import logging
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tiny_llm_pipeline.config import DEFAULT_MODEL, DEFAULT_TRAIN, TrainConfig, lr_at
from tiny_llm_pipeline.data import PackedDataset
from tiny_llm_pipeline.model import TinyLM, save_ckpt, load_ckpt
from tiny_llm_pipeline.tokenizer import SPECIAL_TOKENS, Tok

logger = logging.getLogger(__name__)

_INTERRUPT = False

_DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
_VAL_SAMPLES = 5
_VAL_PREFIX = 32
_VAL_NEW = 32
_EOS_ID = SPECIAL_TOKENS.index("<|eos|>")
_IGNORE_INDEX = -100


def eval_ppl(model: TinyLM, val_bin: Path | str, seq_len: int, device: torch.device) -> float:
    """Token-weighted perplexity on non-overlapping blocks. Empty or short bins return inf."""
    path = Path(val_bin)
    if seq_len < 2 or not path.is_file() or path.stat().st_size == 0:
        return math.inf
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n = int(data.shape[0])
    usable = n - (n % seq_len)
    if usable < seq_len:
        return math.inf
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    try:
        with torch.no_grad():
            for start in range(0, usable, seq_len * 8):
                rows = [
                    np.asarray(data[i : i + seq_len], dtype=np.int64)
                    for i in range(start, min(start + seq_len * 8, usable), seq_len)
                ]
                batch = torch.tensor(np.stack(rows), dtype=torch.long, device=device)
                x, y = batch[:, :-1], _boundary_targets(batch[:, :-1], batch[:, 1:])
                ntok = int((y != _IGNORE_INDEX).sum())
                if ntok == 0:
                    continue
                _, loss = model(x, targets=y)
                total += float(loss) * ntok
                count += ntok
    finally:
        model.train(was_training)
    if count == 0:
        return math.inf
    return math.exp(total / count)


def train_pretrain(
    data_dir: Path | str,
    out_dir: Path | str,
    *,
    tok_hash: str,
    max_steps: int | None,
    max_tokens: int,
    dtype: str = "fp32",
    max_seconds: float | None = None,
    ckpt_every: int = 500,
    seed: int = 42,
    resume: Path | str | None = None,
    train_cfg: TrainConfig = DEFAULT_TRAIN,
    plot_every: int = 0,
    val_every: int = -1,
    tok: Tok | None = None,
) -> Path:
    """Train until `max_steps`, `max_tokens`, or `max_seconds`, then return the ckpt path.

    `max_steps=None` stops only on the token or time budget. The cosine horizon
    is then the number of steps implied by `max_tokens`. `val_every=-1` prints
    nothing from the val set; a positive value prints 5 greedy samples.
    The first Ctrl-C finishes the current step, writes a checkpoint, and returns.
    `resume` continues from that step's data cursor and token count.
    """
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype}")
    if val_every == 0 or val_every < -1:
        raise ValueError("val_every must be -1 or a positive step interval")
    if val_every > 0 and tok is None:
        raise ValueError("val_every requires a tokenizer")
    device = _device()
    source = Path(data_dir)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ckpt_path = dest / "ckpt.pt"
    log_path = dest / "train_log.jsonl"
    train_bin = source / "train.bin"
    if not train_bin.is_file():
        raise FileNotFoundError(f"missing pretrain bin: {train_bin}")

    torch.manual_seed(seed)
    dataset = PackedDataset(train_bin, DEFAULT_MODEL.max_seq_len)
    if len(dataset) == 0:
        raise ValueError(f"pretrain bin is empty: {train_bin}")
    model, opt, step, saved_tokens, saved_cursor = _load_or_init(
        resume, tok_hash, device, train_cfg
    )
    per_step = train_cfg.batch_size * max(1, dataset.seq - 1)
    cursor = step * train_cfg.batch_size if saved_cursor is None else saved_cursor
    tokens = step * per_step if saved_tokens is None else saved_tokens
    elapsed_offset = _truncate_log(log_path, step) if resume is not None else 0.0
    horizon = max_steps if max_steps is not None else max(1, math.ceil(max_tokens / per_step))
    started = time.perf_counter()
    amp = _DTYPES[dtype]
    logger.info(
        "pretrain device=%s batch=%d seq=%d horizon=%d step=%d",
        device.type,
        train_cfg.batch_size,
        dataset.seq,
        horizon,
        step,
    )

    def _save() -> None:
        save_ckpt(
            ckpt_path,
            model,
            model.cfg,
            step,
            tok_hash,
            optimizer=opt,
            seed=seed,
            tokens=tokens,
            cursor=cursor,
        )

    global _INTERRUPT
    _INTERRUPT = False
    previous_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        while max_steps is None or step < max_steps:
            step += 1
            windows = [dataset[cursor + offset] for offset in range(train_cfg.batch_size)]
            cursor += train_cfg.batch_size
            block = torch.stack(windows).to(device)
            x, y = block[:, :-1], _boundary_targets(block[:, :-1], block[:, 1:])
            lr = lr_at(step - 1, horizon, train_cfg)
            for group in opt.param_groups:
                group["lr"] = lr
            try:
                loss = _forward_loss(model, x, y, device, amp)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                raise RuntimeError(
                    f"out of memory on {device.type} with batch_size={train_cfg.batch_size} "
                    f"max_seq_len={model.cfg.max_seq_len}; lower batch_size or max_seq_len"
                ) from exc
            if not torch.isfinite(loss):
                _save()
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            tokens += int(x.numel())
            wall = time.perf_counter() - started
            record: dict[str, float | int] = {
                "step": step,
                "loss": float(loss.detach()),
                "lr": lr,
                "tokens": tokens,
                "elapsed": round(elapsed_offset + wall, 3),
            }
            interrupted = _interrupt_requested()
            timed_out = max_seconds is not None and wall >= max_seconds
            stop = step == max_steps or tokens >= max_tokens or timed_out or interrupted
            if ckpt_every > 0 and (step % ckpt_every == 0 or stop):
                record["val_ppl"] = eval_ppl(
                    model, source / "val.bin", DEFAULT_MODEL.max_seq_len, device
                )
                _save()
            _append_log(log_path, record)
            logger.info(
                "step=%d loss=%.4f lr=%.6g tokens=%d elapsed=%.1fs",
                step,
                record["loss"],
                lr,
                tokens,
                record["elapsed"],
            )
            if plot_every > 0 and (step % plot_every == 0 or stop):
                from tiny_llm_pipeline.viz import write_monitor

                write_monitor(log_path, dest / "monitor.png")
            if val_every > 0 and step % val_every == 0:
                assert tok is not None
                _log_val_samples(model, source / "val.bin", tok, device, step)
            if stop:
                if interrupted:
                    if ckpt_every <= 0:
                        _save()
                    logger.info("saved checkpoint at step %d after interrupt", step)
                elif not ckpt_path.is_file():
                    _save()
                return ckpt_path
        return ckpt_path
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        _INTERRUPT = False


def _boundary_targets(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Ignore the token after `<|eos|>`; it opens the next passage."""
    targets = y.clone()
    targets[x == _EOS_ID] = _IGNORE_INDEX
    return targets


def _passage_ids(val_bin: Path, limit: int) -> list[list[int]]:
    """First `limit` val passages, each without its closing `<|eos|>`."""
    data = np.asarray(np.memmap(val_bin, dtype=np.uint16, mode="r"))
    passages: list[list[int]] = []
    start = 0
    for end in np.flatnonzero(data == _EOS_ID):
        if int(end) > start:
            passages.append(data[start:int(end)].astype(int).tolist())
            if len(passages) == limit:
                break
        start = int(end) + 1
    return passages


def _sample_span(length: int) -> tuple[int, int]:
    """Split a passage in half so gold is a continuation, not the last mark.

    Each side is capped at 32 tokens. A 30-token passage then keeps about 15
    tokens of gold; taking the first 32 as the prompt would leave only `。`.
    """
    prefix_len = min(_VAL_PREFIX, max(1, length // 2))
    return prefix_len, min(_VAL_NEW, length - prefix_len)


def _log_val_samples(
    model: TinyLM, val_bin: Path, tok: Tok, device: torch.device, step: int
) -> None:
    """Greedy continuations for the first 5 val passages, next to the gold text."""
    if not val_bin.is_file() or val_bin.stat().st_size == 0:
        logger.info("step=%d val samples skipped: empty val bin", step)
        return
    passages = [ids for ids in _passage_ids(val_bin, _VAL_SAMPLES) if len(ids) >= 2]
    if not passages:
        logger.info("step=%d val samples skipped: empty val bin", step)
        return
    was_training = model.training
    model.eval()
    try:
        for index, ids in enumerate(passages):
            prefix_len, new_len = _sample_span(len(ids))
            prefix = torch.tensor([ids[:prefix_len]], dtype=torch.long, device=device)
            pred = model.generate(
                prefix,
                new_len,
                temperature=0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
            )
            logger.info("step=%d val %d/%d", step, index + 1, len(passages))
            logger.info("prompt: %s", tok.decode(ids[:prefix_len]))
            logger.info("gold: %s", tok.decode(ids[prefix_len : prefix_len + new_len]))
            logger.info("pred: %s", tok.decode(pred[0, prefix_len:].tolist()))
    finally:
        model.train(was_training)


def _device() -> torch.device:
    """First available accelerator: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _on_sigint(_signum: int, _frame: object) -> None:
    """Finish the current step on the first Ctrl-C. Abort on the second."""
    global _INTERRUPT
    if _INTERRUPT:
        raise KeyboardInterrupt
    _INTERRUPT = True
    logger.info("interrupt: finishing this step, then saving the checkpoint")


def _interrupt_requested() -> bool:
    return _INTERRUPT


def _truncate_log(path: Path, step: int) -> float:
    """Drop log rows past `step`. Return the elapsed time recorded at that step."""
    if not path.is_file():
        return 0.0
    kept: list[str] = []
    elapsed = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["step"]) > step:
            continue
        kept.append(line)
        if int(row["step"]) == step:
            elapsed = float(row["elapsed"])
    path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    return elapsed


def _load_or_init(
    resume: Path | str | None,
    tok_hash: str,
    device: torch.device,
    train_cfg: TrainConfig,
) -> tuple[TinyLM, torch.optim.Optimizer, int, int | None, int | None]:
    if resume is None:
        model = TinyLM(DEFAULT_MODEL).to(device)
        return model, _optimizer(model, train_cfg), 0, None, None
    loaded = load_ckpt(resume, expect_tokenizer_hash=tok_hash)
    model = loaded["model"].to(device)
    opt = _optimizer(model, train_cfg)
    payload = torch.load(Path(resume), map_location="cpu", weights_only=False)
    state = payload.get("optimizer")
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint has no optimizer state: {resume}")
    opt.load_state_dict(state)
    tokens = int(payload["tokens"]) if "tokens" in payload else None
    cursor = int(payload["cursor"]) if "cursor" in payload else None
    return model, opt, int(loaded["step"]), tokens, cursor


def _optimizer(model: TinyLM, train_cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=train_cfg.betas,
    )


def _forward_loss(
    model: TinyLM,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    amp: torch.dtype | None,
) -> torch.Tensor:
    if amp is None:
        _, loss = model(x, targets=y)
    else:
        with torch.autocast(device_type=device.type, dtype=amp):
            _, loss = model(x, targets=y)
    if loss is None:
        raise RuntimeError("pretrain forward returned no loss")
    return loss


def _append_log(path: Path, record: dict[str, float | int]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
