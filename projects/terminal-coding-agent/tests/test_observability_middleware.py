from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from terminal_coding_agent.middleware.observability import ObservabilityMiddleware
from terminal_coding_agent.telemetry.attrs import (
    ATTR_MODEL,
    ATTR_TOOL_NAME,
    assert_no_content_attributes,
)
from terminal_coding_agent.telemetry.setup import setup_tracing


def _model_request() -> dict:
    return {"messages": [], "config": {}}


def test_wrap_model_and_tool_emit_gen_ai_spans(tmp_path: Path) -> None:
    exporter = InMemorySpanExporter()
    setup_tracing(worktree=tmp_path, exporter=exporter)
    middleware = ObservabilityMiddleware(model_name="deepseek-v4-flash")

    def model_handler(request):
        return AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 3,
                "output_tokens": 5,
                "total_tokens": 8,
            },
        )

    middleware.wrap_model_call(_model_request(), model_handler)

    def tool_handler(request):
        return ToolMessage(content="tool ok", tool_call_id="call-1")

    tool_request = SimpleNamespace(
        tool_call={"name": "read_file", "id": "call-1", "args": {"path": "x.py"}}
    )
    middleware.wrap_tool_call(tool_request, tool_handler)

    spans = exporter.get_finished_spans()
    names = {span.name for span in spans}
    assert any(name.startswith("chat ") for name in names)
    assert any(name.startswith("execute_tool ") for name in names)

    chat = next(span for span in spans if span.name.startswith("chat "))
    tool = next(span for span in spans if span.name.startswith("execute_tool "))
    assert chat.attributes[ATTR_MODEL] == "deepseek-v4-flash"
    assert tool.attributes[ATTR_TOOL_NAME] == "read_file"

    for span in spans:
        assert_no_content_attributes(span)


def test_a_span_without_a_model_attribute_is_still_emitted_on_failure(tmp_path: Path) -> None:
    """A chat span is only useful if it says which model failed."""
    exporter = InMemorySpanExporter()
    setup_tracing(worktree=tmp_path, exporter=exporter)
    middleware = ObservabilityMiddleware(model_name="deepseek-v4-flash")

    def boom(request):
        raise RuntimeError("429 rate limited")

    with pytest.raises(RuntimeError):
        middleware.wrap_model_call(_model_request(), boom)

    chat = next(span for span in exporter.get_finished_spans() if span.name.startswith("chat "))
    assert chat.attributes[ATTR_MODEL] == "deepseek-v4-flash"
    assert chat.attributes["error.type"] == "RuntimeError"
    assert_no_content_attributes(chat)
