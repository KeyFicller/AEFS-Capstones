"""Pinned model and training hyperparameters. Single source of truth."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Llama-style decoder-only sizes. Defaults are the pinned ~11M model."""

    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 5
    n_heads: int = 6
    n_kv_heads: int = 2
    ffn_hidden: int = 1024
    max_seq_len: int = 512
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


def lr_at(step: int, max_steps: int, cfg: TrainConfig) -> float:
    """Learning rate at update `step` (0-based) of a `max_steps` cosine schedule."""
    warmup = max(1, int(max_steps * cfg.warmup_ratio))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)
