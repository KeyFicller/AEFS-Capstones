from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from budget import Budget, usage_from_message

from terminal_coding_agent import config
from terminal_coding_agent.state import ToDoItem


def summarize_todos(todo_list: Sequence[ToDoItem]) -> list[dict[str, str]]:
    """Serialize todos for traces: [{description, status}, ...]."""
    return [{"description": item.description, "status": item.status.name} for item in todo_list]


def trace_path(worktree: Path) -> Path:
    """Where the Stop trace is written: {worktree}/.agent/trace.json."""
    return worktree / ".agent" / "trace.json"


def ledger_from_state(state: dict[str, Any]) -> Budget:
    return Budget(
        max_turns=config.MAX_TURNS,
        max_tokens=config.MAX_TOKENS,
        max_cost_rmb=config.MAX_COST_RMB,
        price_cache_hit_per_m=config.PRICE_CACHE_HIT_PER_M,
        price_cache_miss_per_m=config.PRICE_CACHE_MISS_PER_M,
        price_output_per_m=config.PRICE_OUTPUT_PER_M,
        turns=state.get("turns", 0),
        tokens=state.get("tokens", 0),
        input_tokens=state.get("input_tokens", 0),
        output_tokens=state.get("output_tokens", 0),
        cache_read_tokens=state.get("cache_read_tokens", 0),
        cost_rmb=state.get("cost_rmb", 0.0),
        stop_reason=state.get("stop_reason"),
    )


def budget_updates(ledger: Budget) -> dict[str, Any]:
    return {
        "turns": ledger.turns,
        "tokens": ledger.tokens,
        "input_tokens": ledger.input_tokens,
        "output_tokens": ledger.output_tokens,
        "cache_read_tokens": ledger.cache_read_tokens,
        "cost_rmb": ledger.cost_rmb,
        "stop_reason": ledger.stop_reason,
    }


def write_trace(
    path: Path,
    ledger: Budget,
    *,
    todo_list: list[dict[str, str]],
    stop_reason: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "turns": ledger.turns,
        "tokens": ledger.tokens,
        "input_tokens": ledger.input_tokens,
        "output_tokens": ledger.output_tokens,
        "cache_read_tokens": ledger.cache_read_tokens,
        "cost_rmb": ledger.cost_rmb,
        "stop_reason": stop_reason,
        "todo_list": todo_list,
        "finished_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class BudgetSession:
    """with-block helper: enter loads ledger; exit applies queued message usage (no fuse)."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.ledger = ledger_from_state(state)
        self._messages: list[Any] = []

    def __enter__(self) -> "BudgetSession":
        return self

    def observe(self, message: Any) -> None:
        """Queue a model message so exit can call apply_usage. One per model call."""
        if message is not None:
            self._messages.append(message)

    def updates(self) -> dict[str, Any]:
        return budget_updates(self.ledger)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        for message in self._messages:
            input_tokens, output_tokens, cache_read = usage_from_message(message)
            self.ledger.apply_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
            )
        return False
