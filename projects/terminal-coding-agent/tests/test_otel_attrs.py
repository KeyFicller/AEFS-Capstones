from terminal_coding_agent.telemetry.attrs import *
import pytest
from pathlib import Path
from opentelemetry.sdk.trace import TracerProvider

def test_set_chat_attributes(tmp_path: Path):
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("probe") as span:
        set_chat_attributes(span, model="gpt-4o", input_tokens=100, output_tokens=200)
        assert span.attributes[ATTR_MODEL] == "gpt-4o"
        assert span.attributes[ATTR_INPUT_TOKENS] == 100
        assert span.attributes[ATTR_OUTPUT_TOKENS] == 200

def test_set_tool_attributes(tmp_path: Path):
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("probe") as span:
        set_tool_attributes(span, tool_name="tool", tool_call_id="123")
        assert span.attributes[ATTR_TOOL_NAME] == "tool"
        assert span.attributes[ATTR_TOOL_CALL_ID] == "123"

def test_assert_no_content_attributes(tmp_path: Path):
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("probe") as span:
        span.set_attribute("gen_ai.input.messages", "Hello, world!")
        with pytest.raises(ValueError):
            assert_no_content_attributes(span)