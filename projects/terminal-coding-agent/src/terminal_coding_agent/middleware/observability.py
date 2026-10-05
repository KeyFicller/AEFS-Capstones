from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from telemetry import get_tracer
from telemetry.attrs import (
    ATTR_ERROR_TYPE,
    ATTR_OPERATION,
    set_chat_attributes,
    set_tool_attributes,
)


def _ai_message_from_result(result: Any) -> AIMessage | None:
    if isinstance(result, AIMessage):
        return result
    messages = getattr(result, "result", None)
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
    return None


def _usage_tokens(ai_message: AIMessage) -> tuple[int | None, int | None]:
    usage = getattr(ai_message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return None, None
    return usage.get("input_tokens"), usage.get("output_tokens")


class ObservabilityMiddleware(AgentMiddleware):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name

    def wrap_model_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        tracer = get_tracer()
        with tracer.start_as_current_span(f"chat {self.model_name}") as span:
            span.set_attribute(ATTR_OPERATION, "chat")
            try:
                result = handler(request)
            except Exception as exc:
                span.set_attribute(ATTR_ERROR_TYPE, type(exc).__name__)
                set_chat_attributes(span, model=self.model_name)
                raise
            ai_message = _ai_message_from_result(result)
            input_tokens = output_tokens = None
            if ai_message is not None:
                input_tokens, output_tokens = _usage_tokens(ai_message)
            set_chat_attributes(
                span,
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            return result

    async def awrap_model_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        tracer = get_tracer()
        with tracer.start_as_current_span(f"chat {self.model_name}") as span:
            span.set_attribute(ATTR_OPERATION, "chat")
            try:
                result = await handler(request)
            except Exception as exc:
                span.set_attribute(ATTR_ERROR_TYPE, type(exc).__name__)
                set_chat_attributes(span, model=self.model_name)
                raise
            ai_message = _ai_message_from_result(result)
            input_tokens = output_tokens = None
            if ai_message is not None:
                input_tokens, output_tokens = _usage_tokens(ai_message)
            set_chat_attributes(
                span,
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            return result

    def wrap_tool_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        tool_call = getattr(request, "tool_call", None) or {}
        tool_name = str(tool_call.get("name") or "unknown")
        tool_call_id = tool_call.get("id")
        tracer = get_tracer()
        with tracer.start_as_current_span(f"execute_tool {tool_name}") as span:
            span.set_attribute(ATTR_OPERATION, "execute_tool")
            set_tool_attributes(span, tool_name=tool_name, tool_call_id=tool_call_id)
            try:
                result = handler(request)
            except Exception as exc:
                span.set_attribute(ATTR_ERROR_TYPE, type(exc).__name__)
                raise
            if isinstance(result, ToolMessage) and getattr(result, "status", None) == "error":
                span.set_attribute(ATTR_ERROR_TYPE, "tool_error")
            return result

    async def awrap_tool_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        tool_call = getattr(request, "tool_call", None) or {}
        tool_name = str(tool_call.get("name") or "unknown")
        tool_call_id = tool_call.get("id")
        tracer = get_tracer()
        with tracer.start_as_current_span(f"execute_tool {tool_name}") as span:
            span.set_attribute(ATTR_OPERATION, "execute_tool")
            set_tool_attributes(span, tool_name=tool_name, tool_call_id=tool_call_id)
            try:
                result = await handler(request)
            except Exception as exc:
                span.set_attribute(ATTR_ERROR_TYPE, type(exc).__name__)
                raise
            if isinstance(result, ToolMessage) and getattr(result, "status", None) == "error":
                span.set_attribute(ATTR_ERROR_TYPE, "tool_error")
            return result
