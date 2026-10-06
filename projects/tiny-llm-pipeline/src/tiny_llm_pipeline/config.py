"""Pinned model and training hyperparameters. Single source of truth."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Llama-style decoder-only sizes. Defaults follow MiniMind2-Small.

    Vocab, width, depth, and head counts match that model (6400 / 512 / 8 /
    8 query heads / 2 KV heads). `ffn_hidden` uses the current MiniMind
    feed-forward size, ``ceil(d_model * pi / 64) * 64``, which is 1664.
    With the tied embedding counted once that is 28,975,616 parameters.
    `max_seq_len` is the training window recommended for `pretrain_t2t_mini`.
    """

    vocab_size: int = 6400
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_hidden: int = 1664
    max_seq_len: int = 768
    tied_embedding: bool = True
    dropout: float = 0.0


DEFAULT_MODEL = ModelConfig()


@dataclass(frozen=True)
class TrainConfig:
    """Shared optimizer settings for pretrain, SFT, and DPO."""

    lr: float = 1e-3
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_ratio: float = 0.01
    min_lr_ratio: float = 0.1
    batch_size: int = 16


DEFAULT_TRAIN = TrainConfig()

# Hoffmann et al. 2022: compute-optimal pretraining is about 20 tokens per parameter.
CHINCHILLA_TOKENS_PER_PARAM = 20
# 20 * TinyLM(DEFAULT_MODEL).num_params(), with the tied embedding counted once.
PRETRAIN_TOKENS = 579_512_320
# SFT starts from a converged model, so it needs a smaller step than pretraining.
# At 1e-3 the holdout perplexity bottomed at step 4500 and then rose to 36.6.
SFT_LR = 2e-4
# DPO moves an already-finetuned policy toward the preference pairs, so it wants
# a step below SFT's; `DEFAULT_TRAIN.lr` (1e-3, the pretraining value) is far too
# large. The delivered run overrode this with `--lr 3e-5`; see PROJECT.md.
DPO_LR = 1e-5


def lr_at(step: int, max_steps: int, cfg: TrainConfig) -> float:
    """Learning rate at update `step` (0-based) of a `max_steps` cosine schedule."""
    warmup = max(1, int(max_steps * cfg.warmup_ratio))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)
