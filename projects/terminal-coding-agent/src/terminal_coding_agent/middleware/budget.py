import logging
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from terminal_coding_agent.budget import (
    BudgetLedger,
    apply_usage,
    check,
    usage_from_message,
)

logger = logging.getLogger(__name__)


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
    def __init__(self, ledger: BudgetLedger) -> None:
        super().__init__()
        self.ledger = ledger

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        updates: dict[str, Any] = {}
        defaults = {
            "turns": self.ledger.turns,
            "tokens": self.ledger.tokens,
            "input_tokens": self.ledger.input_tokens,
            "output_tokens": self.ledger.output_tokens,
            "cache_read_tokens": self.ledger.cache_read_tokens,
            "cost_rmb": self.ledger.cost_rmb,
        }
        for key, value in defaults.items():
            if key not in state:
                updates[key] = value
        return updates or None

    def _bill(self, result: Any) -> None:
        """Count one model invocation; the cap is only as good as this bookkeeping."""
        ai_message = _ai_message_from_result(result)
        if ai_message is None:
            logger.warning("budget: model result carried no AIMessage; usage not counted")
            return
        input_tokens, output_tokens, cache_read = usage_from_message(ai_message)
        apply_usage(
            self.ledger,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
        )
        if tripped := check(self.ledger):
            self.ledger.stop_reason = tripped

    def wrap_model_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        if reason := check(self.ledger):
            self.ledger.stop_reason = reason
            return AIMessage(content=f"Stopped: budget exceeded ({reason})")

        result = handler(request)
        self._bill(result)
        return result

    async def awrap_model_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        if reason := check(self.ledger):
            self.ledger.stop_reason = reason
            return AIMessage(content=f"Stopped: budget exceeded ({reason})")

        result = await handler(request)
        self._bill(result)
        return result

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        # Stop trace is owned by graph nodes (start_task / end_task), not middleware.
        return None
