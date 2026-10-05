from opentelemetry.trace import Span

ATTR_OPERATION = "gen_ai.operation.name"
ATTR_MODEL = "gen_ai.request.model"
ATTR_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ATTR_TOOL_NAME = "gen_ai.tool.name"
ATTR_TOOL_CALL_ID = "gen_ai.tool.call.id"
ATTR_ERROR_TYPE = "error.type"

FORBIDDEN_ATTR_PREFIXES_OR_KEYS = (
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "gen_ai.prompt",
    "gen_ai.completion",
)


def set_chat_attributes(
    span: Span,
    *,
    model: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    span.set_attribute(ATTR_OPERATION, "chat")
    span.set_attribute(ATTR_MODEL, model)
    if input_tokens is not None:
        span.set_attribute(ATTR_INPUT_TOKENS, input_tokens)
    if output_tokens is not None:
        span.set_attribute(ATTR_OUTPUT_TOKENS, output_tokens)


def set_tool_attributes(
    span: Span,
    *,
    tool_name: str,
    tool_call_id: str | None = None,
) -> None:
    span.set_attribute(ATTR_OPERATION, "execute_tool")
    span.set_attribute(ATTR_TOOL_NAME, tool_name)
    if tool_call_id is not None:
        span.set_attribute(ATTR_TOOL_CALL_ID, tool_call_id)


def assert_no_content_attributes(span: Span) -> None:
    for key in span.attributes:
        if any(key.startswith(prefix) for prefix in FORBIDDEN_ATTR_PREFIXES_OR_KEYS):
            raise ValueError(f"Attribute {key} is forbidden")
