"""Telemetry package exports."""

from terminal_coding_agent.telemetry.chat import chat_span, record_chat_usage, resolve_model_name
from terminal_coding_agent.telemetry.setup import (
    get_tracer,
    otel_jsonl_path,
    setup_tracing,
)

__all__ = [
    "chat_span",
    "get_tracer",
    "otel_jsonl_path",
    "record_chat_usage",
    "resolve_model_name",
    "setup_tracing",
]
