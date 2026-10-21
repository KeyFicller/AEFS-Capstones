"""Direct Preference Optimization against a frozen copy of the SFT checkpoint.

The reference is a second load of the same checkpoint: `eval()` and no grad, and
its file is never written back to. Only the policy goes into AdamW. The loss and
the sequence log-prob follow `design.md` section 6.4 and `impl_plan.md` Task 7.
Checkpoints follow pretraining: save every `ckpt_every` steps and on the first
Ctrl-C.
"""

import hashlib
import logging
import math
import signal
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tiny_llm_pipeline.config import DEFAULT_TRAIN, TrainConfig, lr_at
from tiny_llm_pipeline.data import PreferenceExample, collate_dpo, read_preferences, row_order
from tiny_llm_pipeline.model import TinyLM, load_ckpt, save_ckpt
from tiny_llm_pipeline.tokenizer import Tok
from tiny_llm_pipeline.train import pretrain as _pretrain
from tiny_llm_pipeline.train.pretrain import (
    _append_log,
    _begin_run,
    _check_horizon,
    _device,
    _interrupt_requested,
    _on_sigint,
    _optimizer,
)

logger = logging.getLogger(__name__)

MARGIN_DECAY = 0.9


def dpo_loss(pw: Tensor, pl: Tensor, rw: Tensor, rl: Tensor, beta: float) -> tuple[Tensor, Tensor]:
    """Return `(loss, margin)` for one batch. The caller reduces the batch dim.

    `margin` is `beta * [(logπ_w − logπ_ref_w) − (logπ_l − logπ_ref_l)]`, the
    quantity the run tracks. It is the *gain over the reference*, not an
    outright preference: a positive `margin` can coexist with the policy still
    scoring rejected above chosen. Divide `beta` back out for the raw gap.
    """
    margin = beta * ((pw - rw) - (pl - rl))
    return -F.logsigmoid(margin), margin


def seq_logprob(model, input_ids: Tensor, mask: Tensor) -> Tensor:
    """Sum of completion-token log-probs per row, shaped `[B]`.

    `input_ids` and `mask` are the unshifted chat with right padding. Logits at
    `t` have only seen tokens up to `t`, so `logits[:-1]` predicts `input_ids[1:]`.
    Sum, not mean: the mean is `impl_plan` section 6.4's other option and would
    hide the length term the guardrails look for.
    """
    logits, _ = model(input_ids[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    target = input_ids[:, 1:]
    weights = mask[:, 1:].to(logp.dtype)
    token_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return (token_lp * weights).sum(dim=-1)


def _is_holdout(prompt: str, fraction: float) -> bool:
    """Held-out share keyed by the prompt text.

    `synth-pref` is append-only, so an index split would move a prompt between
    train and holdout whenever the pair file grows. A hash bucket is stable.
    """
    if fraction <= 0.0:
        return False
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 10_000 < fraction * 10_000


def _holdout_readings(
    policy: TinyLM,
    ref: TinyLM,
    rows: list[PreferenceExample],
    tok: Tok,
    max_seq_len: int,
    beta: float,
    device: torch.device,
    batch_size: int = 16,
) -> dict[str, float] | None:
    """Held-out readings, or None when the split is empty.

    `holdout_gap` is the sum-form `logπ_chosen − logπ_rejected` — the quantity
    the loss optimizes, and the one the SFT baseline is quoted against.
    `holdout_win_rate` is the share of pairs with `gap > 0`.

    `holdout_win_rate_norm` scores the same pairs per completion token. A sum
    favours the shorter side, so when the two rates disagree the split's length
    imbalance is doing the work: on the first delivered run the per-pair margin
    correlated with the token-count difference at -0.97, and the normalized win
    rate at step 0 was 10.8% against the sum rate's 22.4%. Reporting both is what
    keeps "DPO got better" from meaning "DPO got shorter".
    """
    if not rows:
        return None
    was_training = policy.training
    policy.eval()
    gaps: list[Tensor] = []
    gaps_norm: list[Tensor] = []
    margins: list[Tensor] = []
    try:
        with torch.no_grad():
            for offset in range(0, len(rows), batch_size):
                batch = collate_dpo(rows[offset : offset + batch_size], tok, max_seq_len)
                chosen = batch["chosen_input_ids"].to(device)
                chosen_mask = batch["chosen_mask"].to(device)
                rejected = batch["rejected_input_ids"].to(device)
                rejected_mask = batch["rejected_mask"].to(device)
                pw = seq_logprob(policy, chosen, chosen_mask)
                pl = seq_logprob(policy, rejected, rejected_mask)
                rw = seq_logprob(ref, chosen, chosen_mask)
                rl = seq_logprob(ref, rejected, rejected_mask)
                gaps.append(pw - pl)
                gaps_norm.append(pw / chosen_mask.sum(-1) - pl / rejected_mask.sum(-1))
                margins.append(dpo_loss(pw, pl, rw, rl, beta)[1])
    finally:
        policy.train(was_training)
    gap = torch.cat(gaps)
    gap_norm = torch.cat(gaps_norm)
    margin = torch.cat(margins)
    return {
        "holdout_gap": float(gap.mean()),
        "holdout_margin": float(margin.mean()),
        "holdout_win_rate": float((gap > 0).float().mean()),
        "holdout_gap_norm": float(gap_norm.mean()),
        "holdout_win_rate_norm": float((gap_norm > 0).float().mean()),
    }


def train_dpo(
    prefs_jsonl: Path | str,
    out_dir: Path | str,
    *,
    ref_ckpt: Path | str,
    tok_hash: str,
    tok: Tok,
    beta: float = 0.1,
    epochs: int = 1,
    max_steps: int | None = None,
    seed: int = 42,
    resume: Path | str | None = None,
    train_cfg: TrainConfig = DEFAULT_TRAIN,
    ckpt_every: int = 500,
    holdout_frac: float = 0.1,
) -> Path:
    """Run DPO from `ref_ckpt` (or `resume`) and return the policy checkpoint.

    `max_steps` wins over `epochs` when set. Pairs whose assistant side has no
    tokens left after truncation are dropped up front and counted in every log
    record, rather than fed in as an all-zero mask.

    `holdout_frac` of the pairs are held out by prompt hash and never trained.
    At every save they give `holdout_gap` (logπ_chosen − logπ_rejected),
    `holdout_margin` (the β-scaled DPO margin) and `holdout_win_rate`, so the
    reported alignment numbers are not in-sample.
    """
    if not 0.0 <= holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in [0, 1), got {holdout_frac}")
    rows = read_preferences(prefs_jsonl)
    if not rows:
        raise ValueError("preference file is empty")

    device = _device()
    ref = load_ckpt(ref_ckpt, expect_tokenizer_hash=tok_hash)["model"].to(device)
    ref.eval()
    ref.requires_grad_(False)
    max_seq_len = ref.cfg.max_seq_len

    rows, dropped = _scorable(rows, tok, max_seq_len)
    if not rows:
        raise ValueError("every preference pair has no assistant tokens to score")
    holdout = [row for row in rows if _is_holdout(row["prompt"], holdout_frac)]
    train_rows = [row for row in rows if not _is_holdout(row["prompt"], holdout_frac)]
    if not train_rows:
        train_rows, holdout = rows, []

    start = ref_ckpt if resume is None else resume
    loaded = load_ckpt(start, expect_tokenizer_hash=tok_hash)
    policy = loaded["model"].to(device)
    opt = _optimizer(policy, train_cfg)
    # `ref_ckpt` is SFT's checkpoint; its step belongs to that run, so a fresh
    # DPO stage counts from 0. Only `resume` continues an existing count.
    step = int(loaded["step"]) if resume is not None else 0
    payload: dict[str, object] | None = None
    if resume is not None:
        payload = torch.load(Path(resume), map_location="cpu", weights_only=False)
        state = payload.get("optimizer")
        if not isinstance(state, dict):
            raise RuntimeError(f"checkpoint has no optimizer state: {resume}")
        opt.load_state_dict(state)
    tokens = int(payload["tokens"]) if payload and "tokens" in payload else 0

    batch_size = train_cfg.batch_size
    order = row_order(len(train_rows), seed)
    per_epoch = max(1, math.ceil(len(train_rows) / batch_size))
    total = max_steps if max_steps is not None else max(1, epochs) * per_epoch
    horizon = max(1, total)

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ckpt_path = dest / "ckpt.pt"
    log_path = dest / "train_log.jsonl"
    horizon = _check_horizon(
        max(1, total), payload.get("horizon") if payload else None, resume
    )
    elapsed_offset = _begin_run(log_path, step, resume=resume is not None)

    torch.manual_seed(seed)
    logger.info(
        "dpo device=%s batch=%d pairs=%d holdout=%d dropped=%d horizon=%d beta=%.3g step=%d",
        device.type,
        batch_size,
        len(train_rows),
        len(holdout),
        dropped,
        horizon,
        beta,
        step,
    )
    holdout_start = _holdout_readings(policy, ref, holdout, tok, max_seq_len, beta, device)
    if holdout_start is not None:
        logger.info(
            "dpo holdout_start gap=%.3f margin=%.3f win_rate=%.3f win_rate_norm=%.3f",
            holdout_start["holdout_gap"],
            holdout_start["holdout_margin"],
            holdout_start["holdout_win_rate"],
            holdout_start["holdout_win_rate_norm"],
        )

    def _save() -> None:
        save_ckpt(
            ckpt_path,
            policy,
            policy.cfg,
            step,
            tok_hash,
            optimizer=opt,
            seed=seed,
            tokens=tokens,
            horizon=horizon,
        )

    margin_avg: float | None = None
    started = time.perf_counter()
    _pretrain._INTERRUPT = False
    previous_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        policy.train()
        while step < total:
            step += 1
            first = ((step - 1) * batch_size) % len(train_rows)
            batch_rows = [train_rows[order[(first + offset) % len(train_rows)]] for offset in range(batch_size)]
            batch = collate_dpo(batch_rows, tok, max_seq_len)
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            lr = lr_at(step - 1, horizon, train_cfg)
            for group in opt.param_groups:
                group["lr"] = lr

            with torch.no_grad():
                ref_w = seq_logprob(ref, chosen_ids, chosen_mask)
                ref_l = seq_logprob(ref, rejected_ids, rejected_mask)
            pol_w = seq_logprob(policy, chosen_ids, chosen_mask)
            pol_l = seq_logprob(policy, rejected_ids, rejected_mask)
            loss, margin = dpo_loss(pol_w, pol_l, ref_w, ref_l, beta)
            loss = loss.mean()
            if not torch.isfinite(loss):
                _save()
                raise RuntimeError(f"non-finite loss at step {step}")

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), train_cfg.grad_clip)
            opt.step()

            tokens += int(chosen_mask[:, 1:].sum() + rejected_mask[:, 1:].sum())
            mean_margin = float(margin.detach().mean())
            margin_avg = (
                mean_margin
                if margin_avg is None
                else MARGIN_DECAY * margin_avg + (1.0 - MARGIN_DECAY) * mean_margin
            )
            record: dict[str, float | int] = {
                "step": step,
                "loss": float(loss.detach()),
                "margin": mean_margin,
                "margin_avg": margin_avg,
                "lr": lr,
                "tokens": tokens,
                "dropped": dropped,
                "elapsed": round(elapsed_offset + time.perf_counter() - started, 3),
            }
            interrupted = _interrupt_requested()
            stop = step == total or interrupted
            if ckpt_every > 0 and (step % ckpt_every == 0 or stop):
                readings = _holdout_readings(
                    policy, ref, holdout, tok, max_seq_len, beta, device
                )
                if readings is not None:
                    record.update(readings)
                _save()
            _append_log(log_path, record)
            logger.info(
                "step=%d loss=%.4f margin=%.4f lr=%.6g tokens=%d",
                step,
                record["loss"],
                mean_margin,
                lr,
                tokens,
            )
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
        _pretrain._INTERRUPT = False


def _scorable(
    rows: list[PreferenceExample], tok: Tok, max_seq_len: int
) -> tuple[list[PreferenceExample], int]:
    """Split off pairs with no assistant tokens left, counting them.

    `_preference_side` truncates from the right, so a prompt that already fills
    the window leaves the completion with nothing to score.
    """
    usable: list[PreferenceExample] = []
    for row in rows:
        batch = collate_dpo([row], tok, max_seq_len)
        if int(batch["chosen_mask"][:, 1:].sum()) and int(batch["rejected_mask"][:, 1:].sum()):
            usable.append(row)
    return usable, len(rows) - len(usable)
