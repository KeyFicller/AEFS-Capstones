from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry.trace import Span

from terminal_coding_agent.telemetry.attrs import set_chat_attributes
from terminal_coding_agent.telemetry.setup import get_tracer


def resolve_model_name(model: Any) -> str:
    """Prefer an explicit string; otherwise read LangChain chat model fields."""
    if isinstance(model, str):
        return model
    for attr in ("model_name", "model"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


def record_chat_usage(span: Span, message: Any, *, model: str) -> None:
    usage = getattr(message, "usage_metadata", None) or {}
    set_chat_attributes(
        span,
        model=model,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )


@contextmanager
def chat_span(model: Any) -> Iterator[Span]:
    model_name = resolve_model_name(model)
    tracer = get_tracer()
    with tracer.start_as_current_span(f"chat {model_name}") as span:
        set_chat_attributes(span, model=model_name)
        yield span
