import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from terminal_coding_agent import config
from terminal_coding_agent.state import ToDoItem


@dataclass
class BudgetLedger:
    turns: int = 0
    tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_rmb: float = 0.0
    stop_reason: str | None = None


def summarize_todos(todo_list: Sequence[ToDoItem]) -> list[dict[str, str]]:
    """Serialize todos for traces: [{description, status}, ...]."""
    return [{"description": item.description, "status": item.status.name} for item in todo_list]


def trace_path(worktree: Path) -> Path:
    """Where the Stop trace is written: {worktree}/.agent/trace.json."""
    return worktree / ".agent" / "trace.json"


def price_usage(*, input_tokens: int, output_tokens: int, cache_read_tokens: int) -> float:
    cache_read_tokens = min(cache_read_tokens, input_tokens)
    return (
        cache_read_tokens / 1e6 * config.PRICE_CACHE_HIT_PER_M
        + (input_tokens - cache_read_tokens) / 1e6 * config.PRICE_CACHE_MISS_PER_M
        + output_tokens / 1e6 * config.PRICE_OUTPUT_PER_M
    )


def check(ledger: BudgetLedger) -> str | None:
    if ledger.turns >= config.MAX_TURNS:
        return "max_turns"
    if ledger.tokens >= config.MAX_TOKENS:
        return "max_tokens"
    if ledger.cost_rmb >= config.MAX_COST_RMB:
        return "max_cost"
    return None


def apply_usage(
    ledger: BudgetLedger,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> None:
    cache_read_tokens = min(cache_read_tokens, input_tokens)
    ledger.turns += 1
    ledger.input_tokens += input_tokens
    ledger.output_tokens += output_tokens
    ledger.cache_read_tokens += cache_read_tokens
    ledger.tokens += input_tokens + output_tokens
    ledger.cost_rmb += price_usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
    )


def write_trace(
    path: Path,
    ledger: BudgetLedger,
    *,
    todo_list: list[dict[str, str]],
    stop_reason: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **asdict(ledger),
        "stop_reason": stop_reason,
        "todo_list": todo_list,
        "finished_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ledger_from_state(state: dict[str, Any]) -> BudgetLedger:
    return BudgetLedger(
        turns=state.get("turns", 0),
        tokens=state.get("tokens", 0),
        input_tokens=state.get("input_tokens", 0),
        output_tokens=state.get("output_tokens", 0),
        cache_read_tokens=state.get("cache_read_tokens", 0),
        cost_rmb=state.get("cost_rmb", 0.0),
        stop_reason=state.get("stop_reason"),
    )


def budget_updates(ledger: BudgetLedger) -> dict[str, Any]:
    return {
        "turns": ledger.turns,
        "tokens": ledger.tokens,
        "input_tokens": ledger.input_tokens,
        "output_tokens": ledger.output_tokens,
        "cache_read_tokens": ledger.cache_read_tokens,
        "cost_rmb": ledger.cost_rmb,
        "stop_reason": ledger.stop_reason,
    }


def usage_from_message(message: Any) -> tuple[int, int, int]:
    meta = getattr(message, "usage_metadata", None) or {}
    input_tokens = int(meta.get("input_tokens") or 0)
    output_tokens = int(meta.get("output_tokens") or 0)
    details = meta.get("input_token_details") or {}
    cache_read = int(details.get("cache_read") or meta.get("cache_read_tokens") or 0)
    return input_tokens, output_tokens, cache_read


class BudgetSession:
    """with-block helper: enter loads ledger; exit applies queued message usage (no fuse)."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.ledger = ledger_from_state(state)
        self._messages: list[Any] = []

    def __enter__(self) -> BudgetSession:
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
            apply_usage(
                self.ledger,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
            )
        return False
