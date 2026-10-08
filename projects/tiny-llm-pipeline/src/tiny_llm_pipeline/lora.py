"""Low-rank adapters on the attention projections, applied and folded at load.

`inject_lora` swaps each attention projection for a `LoRALinear` and freezes
everything else. `b` starts at zero, so a freshly injected model returns exactly
the base model's output. `merge_lora` folds `scale * B @ A` back into the weight
and restores plain `nn.Linear`, which is what a checkpoint's consumers get.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ATTN_TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class LoRAConfig:
    """Adapter shape. `rank <= 0` means no adapter."""

    rank: int = 8
    alpha: int = 16
    targets: tuple[str, ...] = ATTN_TARGETS

    @property
    def enabled(self) -> bool:
        """Whether this config adds any trainable parameters."""
        return self.rank > 0


class LoRALinear(nn.Linear):
    """A frozen `nn.Linear` plus a low-rank update, zero-initialized on `b`."""

    def __init__(self, base: nn.Linear, cfg: LoRAConfig) -> None:
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None)
        self.weight = base.weight
        if base.bias is not None:
            self.bias = base.bias
        self.r = cfg.rank
        self.scale = cfg.alpha / cfg.rank
        self.a = nn.Parameter(torch.empty(cfg.rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, cfg.rank))
        nn.init.normal_(self.a, std=1.0 / math.sqrt(base.in_features))
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(x, self.weight, self.bias)
        return out + self.scale * ((x @ self.a.T) @ self.b.T)


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> int:
    """Swap attention projections for `LoRALinear`, freeze the rest.

    Returns the number of trainable parameters, zero when `cfg` is disabled.
    """
    if not cfg.enabled:
        return 0
    swaps = [
        (parent, name)
        for parent in list(model.modules())
        for name in cfg.targets
        if isinstance(getattr(parent, name, None), nn.Linear)
        and not isinstance(getattr(parent, name), LoRALinear)
    ]
    for parent, name in swaps:
        setattr(parent, name, LoRALinear(getattr(parent, name), cfg))
    model.requires_grad_(False)
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.a.requires_grad_(True)
            module.b.requires_grad_(True)
            count += module.a.numel() + module.b.numel()
    return count


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold `scale * B @ A` into the weight and restore plain `nn.Linear`."""
    swaps = [
        (parent, name, child)
        for parent in list(model.modules())
        for name, child in list(parent.named_children())
        if isinstance(child, LoRALinear)
    ]
    for parent, name, child in swaps:
        merged = nn.Linear(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            device=child.weight.device,
            dtype=child.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(child.weight + child.scale * (child.b @ child.a))
            if child.bias is not None:
                merged.bias.copy_(child.bias)
        setattr(parent, name, merged)
    return model


def adapter_state_dict(model: nn.Module) -> dict[str, dict[str, Tensor]]:
    """`{qualified layer name: {"a": ..., "b": ...}}` for every `LoRALinear`."""
    return {
        name: {"a": module.a.detach().clone(), "b": module.b.detach().clone()}
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }


def load_adapter_state(model: nn.Module, adapter: dict[str, dict[str, Tensor]]) -> None:
    """Copy `adapter` into an already-injected model. Layer names must match."""
    modules = {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}
    if set(modules) != set(adapter):
        difference = sorted(set(adapter) ^ set(modules))
        raise ValueError(f"adapter layers do not match the model: {difference}")
    with torch.no_grad():
        for name, module in modules.items():
            module.a.copy_(adapter[name]["a"])
            module.b.copy_(adapter[name]["b"])
