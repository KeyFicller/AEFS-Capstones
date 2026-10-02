"""Tool-log middleware: forward every executed tool call to an injected renderer."""

from collections.abc import Callable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage


def _content(result: Any) -> str:
    """The tool's textual result, whichever shape the handler returned."""
    if isinstance(result, ToolMessage):
        content = result.content
        return content if isinstance(content, str) else str(content)
    return str(result)


class ToolLogMiddleware(AgentMiddleware):
    """Emits after the tool ran, so the renderer sees the real output.

    Kept innermost on purpose: a call blocked by an earlier guard (e.g. the
    destructive-command check in `SafetyMiddleware`) never reaches the handler and
    is therefore not logged — a policy refusal is not tool output. Without an
    injected renderer this middleware is not installed at all, so Harbor's path is
    unchanged.
    """

    def __init__(self, render: Callable[[str, Mapping[str, Any], str], None]) -> None:
        super().__init__()
        self._render = render

    def _emit(self, request: Any, result: Any) -> None:
        tool_call = getattr(request, "tool_call", None) or {}
        self._render(
            str(tool_call.get("name") or "unknown"),
            tool_call.get("args") or {},
            _content(result),
        )

    def wrap_tool_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        result = handler(request)
        self._emit(request, result)
        return result

    async def awrap_tool_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        result = await handler(request)
        self._emit(request, result)
        return result
