"""Per-ask spend caps.

Calls are counted at the node boundary. Tokens come only from ``BudgetCallback``.
Seconds start when the budget is created, so a slow model load counts against the ask.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from multimodal_doc_qa.config import Settings


@dataclass
class Budget:
    """Caps and spend for one ask. ``None`` on a cap means that cap is off."""

    max_calls: int | None
    max_tokens: int | None
    max_seconds: float | None
    calls: int = 0
    tokens: int = 0
    started: float = field(default_factory=time.perf_counter)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Budget":
        return cls(
            max_calls=settings.max_ask_calls,
            max_tokens=settings.max_ask_tokens,
            max_seconds=settings.max_ask_seconds,
        )

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def exhausted(self) -> bool:
        """Whether any cap is already spent. Checked before reserving the next call, so a cap of ``n`` still allows ``n`` calls."""
        return (
            (self.max_calls is not None and self.calls >= self.max_calls)
            or (self.max_tokens is not None and self.tokens >= self.max_tokens)
            or (self.max_seconds is not None and self.elapsed() >= self.max_seconds)
        )

    def spend_call(self) -> None:
        self.calls += 1

    def callback(self) -> BaseCallbackHandler:
        """Handler that adds provider token usage to this budget. Unattached, the token cap stays at zero."""
        return BudgetCallback(self)


class BudgetCallback(BaseCallbackHandler):
    """Add each finished call's ``total_tokens`` to a ``Budget``."""

    def __init__(self, budget: Budget) -> None:
        super().__init__()
        self.budget = budget

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for batch in getattr(response, "generations", []) or []:
            for generation in batch:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if not usage:
                    continue
                # Some providers report only the split, never `total_tokens`; defaulting
                # that to 0 would silently defeat the token cap.
                total = usage.get("total_tokens")
                if total is None:
                    total = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
                self.budget.tokens += int(total or 0)
