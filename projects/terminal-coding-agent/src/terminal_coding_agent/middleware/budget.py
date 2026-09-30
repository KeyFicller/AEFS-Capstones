from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from terminal_coding_agent.budget import (
    BudgetLedger,
    apply_usage,
    check,
    usage_from_message,
    write_trace,
)


def _ai_message_from_result(result: Any) -> AIMessage | None:
    """Normalize wrap_model_call handler output to an AIMessage for usage extraction."""
    if isinstance(result, AIMessage):
        return result
    messages = getattr(result, "result", None)
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
    return None


class BudgetMiddleware(AgentMiddleware):
    def __init__(
        self,
        ledger: BudgetLedger,
        *,
        trace_path: Path | None = None,
        todo_list: list[dict[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.ledger = ledger
        self.trace_path = trace_path
        self.todo_list = todo_list or []

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        updates: dict[str, Any] = {}
        defaults = {
            "turns": self.ledger.turns,
            "tokens": self.ledger.tokens,
            "input_tokens": self.ledger.input_tokens,
            "output_tokens": self.ledger.output_tokens,
            "cache_read_tokens": self.ledger.cache_read_tokens,
            "cost_usd": self.ledger.cost_usd,
        }
        for key, value in defaults.items():
            if key not in state:
                updates[key] = value
        return updates or None

    def wrap_model_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        reason = check(self.ledger)
        if reason is not None:
            self.ledger.stop_reason = reason
            return AIMessage(content=f"Stopped: budget exceeded ({reason})")

        result = handler(request)
        ai_message = _ai_message_from_result(result)
        if ai_message is not None:
            input_tokens, output_tokens, cache_read = usage_from_message(ai_message)
            apply_usage(
                self.ledger,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
            )
            tripped = check(self.ledger)
            if tripped is not None:
                self.ledger.stop_reason = tripped

        return result

    async def awrap_model_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        reason = check(self.ledger)
        if reason is not None:
            self.ledger.stop_reason = reason
            return AIMessage(content=f"Stopped: budget exceeded ({reason})")

        result = await handler(request)
        ai_message = _ai_message_from_result(result)
        if ai_message is not None:
            input_tokens, output_tokens, cache_read = usage_from_message(ai_message)
            apply_usage(
                self.ledger,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
            )
            tripped = check(self.ledger)
            if tripped is not None:
                self.ledger.stop_reason = tripped

        return result

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if self.trace_path is not None:
            write_trace(
                self.trace_path,
                self.ledger,
                todo_list=self.todo_list,
                stop_reason=self.ledger.stop_reason or "completed",
            )
        return None
