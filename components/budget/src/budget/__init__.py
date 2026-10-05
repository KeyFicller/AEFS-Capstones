"""Optional spend caps. Callers decide which entry to use."""

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


def usage_from_message(message: object) -> tuple[int, int, int]:
    """Return input, output, and cache-read tokens from ``usage_metadata``."""
    meta = getattr(message, "usage_metadata", None) or {}
    input_tokens = int(meta.get("input_tokens") or 0)
    output_tokens = int(meta.get("output_tokens") or 0)
    details = meta.get("input_token_details") or {}
    cache_read = int(details.get("cache_read") or meta.get("cache_read_tokens") or 0)
    return input_tokens, output_tokens, cache_read


@dataclass
class Budget:
    """Caps and spend. A ``None`` cap is off. Prices default to zero."""

    max_calls: int | None = None
    max_turns: int | None = None
    max_tokens: int | None = None
    max_cost_rmb: float | None = None
    max_seconds: float | None = None

    calls: int = 0
    turns: int = 0
    tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_rmb: float = 0.0
    stop_reason: str | None = None

    price_cache_hit_per_m: float = 0.0
    price_cache_miss_per_m: float = 0.0
    price_output_per_m: float = 0.0

    started: float = field(default_factory=time.perf_counter)

    def spend_call(self) -> None:
        self.calls += 1

    def apply_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
    ) -> None:
        cache_read_tokens = min(cache_read_tokens, input_tokens)
        self.turns += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_read_tokens += cache_read_tokens
        self.tokens += input_tokens + output_tokens
        self.cost_rmb += (
            cache_read_tokens / 1e6 * self.price_cache_hit_per_m
            + (input_tokens - cache_read_tokens) / 1e6 * self.price_cache_miss_per_m
            + output_tokens / 1e6 * self.price_output_per_m
        )

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def check(self) -> str | None:
        if self.max_calls is not None and self.calls >= self.max_calls:
            return "max_calls"
        if self.max_turns is not None and self.turns >= self.max_turns:
            return "max_turns"
        if self.max_tokens is not None and self.tokens >= self.max_tokens:
            return "max_tokens"
        if self.max_cost_rmb is not None and self.cost_rmb >= self.max_cost_rmb:
            return "max_cost"
        if self.max_seconds is not None and self.elapsed() >= self.max_seconds:
            return "max_seconds"
        return None

    def exhausted(self) -> bool:
        return self.check() is not None

    def callback(self) -> "BudgetCallback":
        return BudgetCallback(self)


class BudgetCallback(BaseCallbackHandler):
    """Add provider token totals. Does not count a call or a turn."""

    def __init__(self, budget: Budget) -> None:
        super().__init__()
        self.budget = budget

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for batch in getattr(response, "generations", []) or []:
            for generation in batch:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if not usage:
                    continue
                total = usage.get("total_tokens")
                if total is None:
                    total = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
                self.budget.tokens += int(total or 0)
