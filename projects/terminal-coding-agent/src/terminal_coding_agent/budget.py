from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    cost_usd: float = 0.0
    stop_reason: str | None = None


def summarize_todos(todo_list: Sequence[ToDoItem]) -> list[dict[str, str]]:
    """Serialize todos for traces: [{description, status}, ...]."""
    return [
        {"description": item.description, "status": item.status.name}
        for item in todo_list
    ]


def price_usage(*, input_tokens: int, output_tokens: int, cache_read_tokens: int) -> float:
    cache_read_tokens = min(cache_read_tokens, input_tokens)
    return (
        cache_read_tokens / 1e6 * config.PRICE_CACHE_HIT_PER_M
        + (input_tokens - cache_read_tokens) / 1e6 * config.PRICE_CACHE_MISS_PER_M
        + output_tokens / 1e6 * config.PRICE_OUTPUT_PER_M
    )


def check(ledger: BudgetLedger) -> str | None:
    # tokens fuse uses cumulative ledger.tokens as the 200k proxy
    if ledger.turns >= config.MAX_TURNS:
        return "max_turns"
    if ledger.tokens >= config.MAX_CONTEXT_TOKENS:
        return "max_tokens"
    if ledger.cost_usd >= config.MAX_COST_USD:
        return "max_cost"
    return None


def apply_usage(
    ledger: BudgetLedger,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> None:
    ledger.turns += 1
    ledger.input_tokens += input_tokens
    ledger.output_tokens += output_tokens
    ledger.cache_read_tokens += cache_read_tokens
    ledger.tokens += input_tokens + output_tokens
    ledger.cost_usd += price_usage(
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
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ledger_from_state(state: dict[str, Any]) -> BudgetLedger:
    return BudgetLedger(
        turns=state.get("turns", 0),
        tokens=state.get("tokens", 0),
        input_tokens=state.get("input_tokens", 0),
        output_tokens=state.get("output_tokens", 0),
        cache_read_tokens=state.get("cache_read_tokens", 0),
        cost_usd=state.get("cost_usd", 0.0),
        stop_reason=state.get("stop_reason"),
    )


def budget_updates(ledger: BudgetLedger) -> dict[str, Any]:
    return {
        "turns": ledger.turns,
        "tokens": ledger.tokens,
        "input_tokens": ledger.input_tokens,
        "output_tokens": ledger.output_tokens,
        "cache_read_tokens": ledger.cache_read_tokens,
        "cost_usd": ledger.cost_usd,
        "stop_reason": ledger.stop_reason,
    }


def usage_from_message(message: Any) -> tuple[int, int, int]:
    meta = getattr(message, "usage_metadata", None) or {}
    input_tokens = int(meta.get("input_tokens") or 0)
    output_tokens = int(meta.get("output_tokens") or 0)
    details = meta.get("input_token_details") or {}
    cache_read = int(details.get("cache_read") or meta.get("cache_read_tokens") or 0)
    return input_tokens, output_tokens, cache_read
