"""Decoder-only Llama-style language model and checkpoint I/O.

Pre-norm RMSNorm, RoPE, grouped-query attention, and SwiGLU, with no bias.
`nn.RMSNorm` and `scaled_dot_product_attention` come from torch. RoPE and the
SwiGLU linear trio do not have a torch module.
"""

import dataclasses
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tiny_llm_pipeline.config import ModelConfig

_ROPE_THETA = 10_000.0
_RMS_EPS = 1e-5


def rope_cache(
    seq: int, head_dim: int, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Cos/sin tables for rotate-half RoPE, shaped `[1, 1, T, head_dim]`."""
    inv = 1.0 / (
        _ROPE_THETA ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    freqs = torch.outer(torch.arange(seq, device=device).float(), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate q and k. Both are `[B, H, T, head_dim]`."""

    def rotate_half(x: Tensor) -> Tensor:
        half = x.size(-1) // 2
        a, b = x[..., :half], x[..., half:]
        return torch.cat((-b, a), dim=-1)

    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    """Grouped-query attention. The causal mask and KV repeat live in SDPA."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model {cfg.d_model} is not divisible by n_heads {cfg.n_heads}")
        if cfg.n_heads % cfg.n_kv_heads != 0:
            raise ValueError(
                f"n_heads {cfg.n_heads} is not divisible by n_kv_heads {cfg.n_kv_heads}"
            )
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = rope_cache(seq, self.head_dim, x.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True, dropout_p=0.0
        )
        merged = attn.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.o_proj(merged)


class SwiGLU(nn.Module):
    """Three bias-free linears. torch has `silu` but not this combination."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    """Pre-norm residual block."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model, eps=_RMS_EPS)
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model, eps=_RMS_EPS)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyLM(nn.Module):
    """Llama-style decoder-only Transformer configured by `ModelConfig`."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = nn.RMSNorm(cfg.d_model, eps=_RMS_EPS)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tied_embedding:
            self.lm_head.weight = self.tok_emb.weight
        # Default Embedding init is N(0, 1). With a tied head the residual of the
        # current token saturates the logit, so the smoke loss starts at 0.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        if not cfg.tied_embedding:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Return the parameter count, counting a tied embedding once."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Return logits `[B, T, V]` and optional token cross-entropy.

        Logits stay aligned with `idx`. The caller shifts for next-token loss.
        `ignore_index=-100`. `targets is None` means no loss.
        """
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> Tensor:
        """Append sampled tokens to `idx`.

        Filters run in order: repetition penalty (divide), top-k, top-p.
        `temperature <= 0` is argmax; otherwise softmax and multinomial.
        """
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.max_seq_len :]
            logits = self.forward(ctx)[0][:, -1, :]
            logits = _filter_logits(logits, idx, top_k, top_p, repetition_penalty)
            if temperature <= 0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx


def _filter_logits(
    logits: Tensor,
    seen: Tensor,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> Tensor:
    if repetition_penalty != 1.0:
        for row in range(logits.size(0)):
            ids = torch.unique(seen[row])
            score = logits[row, ids]
            # Dividing alone would move negative scores toward zero, which
            # raises the tokens it is meant to suppress. Match CTRL/HF.
            logits[row, ids] = torch.where(
                score < 0, score * repetition_penalty, score / repetition_penalty
            )
    if top_k > 0:
        kth = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0.0 < top_p < 1.0:
        ordered, index = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(torch.softmax(ordered, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        ordered = ordered.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, index, ordered)
    return logits


def save_ckpt(
    path: Path | str,
    model: TinyLM,
    cfg: ModelConfig,
    step: int,
    tokenizer_hash: str,
    optimizer: torch.optim.Optimizer | None = None,
    seed: int = 42,
    *,
    tokens: int | None = None,
    cursor: int | None = None,
) -> None:
    """Write a checkpoint, creating parent directories.

    Persist weights, `cfg`, `step`, `seed`, and `tokenizer_hash`. Include
    optimizer state when `optimizer` is given, and `tokens` / `cursor` when
    set. Replace the destination only after the temporary file is written.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "cfg": dataclasses.asdict(cfg),
        "step": step,
        "seed": seed,
        "tokenizer_hash": tokenizer_hash,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if tokens is not None:
        payload["tokens"] = tokens
    if cursor is not None:
        payload["cursor"] = cursor
    temporary = dest.with_suffix(dest.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(dest)


class LoadedCheckpoint(TypedDict):
    """What `load_ckpt` returns."""

    model: TinyLM
    cfg: ModelConfig
    step: int
    seed: int
    tokenizer_hash: str


def load_ckpt(
    path: Path | str, expect_tokenizer_hash: str | None = None
) -> LoadedCheckpoint:
    """Load a checkpoint.

    Raise `ValueError` when `expect_tokenizer_hash` is set and does not match
    the file.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    got = str(payload["tokenizer_hash"])
    if expect_tokenizer_hash is not None and got != expect_tokenizer_hash:
        raise ValueError(
            f"tokenizer hash mismatch: checkpoint has {got}, expected {expect_tokenizer_hash}"
        )
    cfg = ModelConfig(**payload["cfg"])
    model = TinyLM(cfg)
    model.load_state_dict(payload["model"])
    return {
        "model": model,
        "cfg": cfg,
        "step": int(payload["step"]),
        "seed": int(payload["seed"]),
        "tokenizer_hash": got,
    }
