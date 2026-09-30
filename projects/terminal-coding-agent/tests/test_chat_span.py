from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from terminal_coding_agent.telemetry.attrs import ATTR_INPUT_TOKENS, ATTR_MODEL
from terminal_coding_agent.telemetry.chat import (
    chat_span,
    record_chat_usage,
    resolve_model_name,
)
from terminal_coding_agent.telemetry.setup import setup_tracing


def test_resolve_model_name_from_chat_model() -> None:
    model = SimpleNamespace(model_name="deepseek-v4-flash", name=None)
    assert resolve_model_name(model) == "deepseek-v4-flash"
    assert resolve_model_name("explicit") == "explicit"


def test_chat_span_records_attributes(tmp_path: Path) -> None:
    exporter = InMemorySpanExporter()
    setup_tracing(worktree=tmp_path, exporter=exporter)
    with chat_span("deepseek-v4-flash") as span:
        record_chat_usage(
            span,
            SimpleNamespace(usage_metadata={"input_tokens": 7, "output_tokens": 2}),
            model="deepseek-v4-flash",
        )

    spans = exporter.get_finished_spans()
    assert any(s.name == "chat deepseek-v4-flash" for s in spans)
    chat = next(s for s in spans if s.name.startswith("chat "))
    assert chat.attributes[ATTR_MODEL] == "deepseek-v4-flash"
    assert chat.attributes[ATTR_INPUT_TOKENS] == 7
