import pytest
from intent import Classifier, IntentVerdict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class _StubModel:
    """Records the bound schema and the messages, and returns a canned raw+parsed pair."""

    def __init__(self, intent: str) -> None:
        self.intent = intent
        self.schema = None
        self.messages = None

    def with_structured_output(self, schema, include_raw=False):  # noqa: ARG002
        self.schema = schema
        outer = self

        class _Structured:
            def invoke(self, messages):
                outer.messages = list(messages)
                raw = AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                )
                return {
                    "raw": raw,
                    "parsed": IntentVerdict(intent=outer.intent),
                    "parsing_error": None,
                }

        return _Structured()


def test_classify_returns_the_verdict_and_the_raw_message() -> None:
    model = _StubModel("chat")

    result = Classifier(model, work_hint="requires retrieval").classify(
        [HumanMessage(content="hi")]
    )

    assert result.intent == "chat"
    assert result.raw.usage_metadata["total_tokens"] == 3
    assert model.schema is IntentVerdict


def test_the_work_hint_is_embedded_in_the_system_prompt() -> None:
    model = _StubModel("work")

    Classifier(model, work_hint="requires retrieving the corpus").classify(
        [HumanMessage(content="?")]
    )

    assert "requires retrieving the corpus" in model.messages[0].content
    assert model.messages[0].type == "system"
    assert model.messages[-1].content == "User: ?"


def test_a_parse_failure_raises_so_the_caller_can_degrade() -> None:
    class _Broken:
        def with_structured_output(self, schema, include_raw=False):  # noqa: ARG002
            class _Structured:
                def invoke(self, messages):
                    return {"raw": None, "parsed": None, "parsing_error": ValueError("nope")}

            return _Structured()

    with pytest.raises(ValueError):
        Classifier(_Broken(), work_hint="h").classify([HumanMessage(content="?")])


def _payload(model: _StubModel) -> str:
    return model.messages[-1].content


def test_only_the_trailing_window_reaches_the_model() -> None:
    model = _StubModel("work")

    Classifier(model, work_hint="h", window=3).classify(
        [HumanMessage(content=f"t{i}") for i in range(5)]
    )

    payload = _payload(model)
    assert "t4" in payload and "t3" in payload and "t2" in payload
    assert "t1" not in payload and "t0" not in payload


def test_tool_and_empty_turns_do_not_consume_the_window() -> None:
    model = _StubModel("work")
    messages = [
        HumanMessage(content="old"),
        AIMessage(content="", tool_calls=[{"id": "1", "name": "edit_file", "args": {}}]),
        ToolMessage(content="x" * 50, tool_call_id="1"),
        AIMessage(content="mid"),
        HumanMessage(content="new"),
    ]

    Classifier(model, work_hint="h", window=2).classify(messages)

    payload = _payload(model)
    assert "mid" in payload and "new" in payload
    assert "old" not in payload
    assert "edit_file" not in payload


def test_the_payload_carries_no_tool_call() -> None:
    model = _StubModel("work")
    messages = [
        AIMessage(content="done", tool_calls=[{"id": "1", "name": "run_shell", "args": {}}]),
        HumanMessage(content="again"),
    ]

    Classifier(model, work_hint="h").classify(messages)

    assert "run_shell" not in _payload(model)
    assert getattr(model.messages[-1], "tool_calls", None) in (None, [])


def test_a_long_turn_is_truncated_with_a_marker() -> None:
    model = _StubModel("chat")

    Classifier(model, work_hint="h", per_turn_chars=10).classify([AIMessage(content="x" * 100)])

    payload = _payload(model)
    assert payload.endswith("...")
    assert len(payload) <= len("Assistant: ") + 10 + 3
    assert "x" * 100 not in payload


def test_a_turn_with_no_text_raises_so_the_caller_degrades() -> None:
    model = _StubModel("chat")

    with pytest.raises(ValueError):
        Classifier(model, work_hint="h").classify([AIMessage(content="")])
