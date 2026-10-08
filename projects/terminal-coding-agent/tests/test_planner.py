from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from terminal_coding_agent.models import AgentModels
from terminal_coding_agent.planner import build_planner


class _StructuredStub:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def invoke(self, input, config=None):  # noqa: A002 - mirrors BaseChatModel
        raise self._exc


class _PlannerStub:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def with_structured_output(self, schema, include_raw=False):
        return _StructuredStub(self._exc)


class _RawResponseStub:
    """What `with_structured_output(include_raw=True)` returns on a parse failure."""

    def __init__(self, response: dict) -> None:
        self._response = response

    def invoke(self, input, config=None):  # noqa: A002 - mirrors BaseChatModel
        return self._response


class _RawPlannerStub:
    def __init__(self, response: dict) -> None:
        self._response = response

    def with_structured_output(self, schema, include_raw=False):
        return _RawResponseStub(self._response)


def test_make_plan_converts_model_crash_into_stop_reason() -> None:
    model = _PlannerStub(RuntimeError("400"))
    node = build_planner(AgentModels(planner=model, executor=model))

    out = node({"messages": []}, RunnableConfig())

    assert out["stop_reason"] == "planner_error:RuntimeError"


def test_make_plan_reports_a_structured_output_parse_failure() -> None:
    """`parsed=None` + `parsing_error` must be diagnosable, not an AttributeError."""
    raw = AIMessage(
        content="not json",
        usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )
    model = _RawPlannerStub(
        {"raw": raw, "parsed": None, "parsing_error": ValueError("no plan in output")}
    )
    node = build_planner(AgentModels(planner=model, executor=model))

    out = node({"messages": []}, RunnableConfig())

    assert out["stop_reason"] == "planner_error:ValueError"
    assert out["turns"] == 1, "the billed call is still counted"


class _SchemaStub:
    """Records the schema the planner asked for and returns a canned parsed plan."""

    def __init__(self, parsed) -> None:
        self.parsed = parsed
        self.schema = None

    def with_structured_output(self, schema, include_raw=False):  # noqa: ARG002
        self.schema = schema
        outer = self

        class _Structured:
            def invoke(self, messages, config=None):  # noqa: ARG002
                raw = AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                return {"raw": raw, "parsed": outer.parsed, "parsing_error": None}

        return _Structured()


def test_the_planner_always_uses_the_plain_plan_schema() -> None:
    from terminal_coding_agent.state import Plan

    parsed = Plan(task="fix the typo", steps=["locate it", "edit it"])
    planner = _SchemaStub(parsed)
    node = build_planner(AgentModels(planner=planner, executor=planner))

    out = node({"messages": []}, RunnableConfig())

    assert planner.schema is Plan
    assert [item.description for item in out["todo_list"]] == ["locate it", "edit it"]


def test_the_planner_prompt_is_injected_only_on_request() -> None:
    """Harbor parity: without the flag the planner call is exactly as before."""
    from terminal_coding_agent.state import Plan

    class _Capturing:
        def __init__(self) -> None:
            self.seen: list = []

        def with_structured_output(self, schema, include_raw=False):  # noqa: ARG002
            outer = self

            class _Structured:
                def invoke(self, messages, config=None):  # noqa: ARG002
                    outer.seen = list(messages)
                    raw = AIMessage(
                        content="",
                        usage_metadata={"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
                    )
                    return {
                        "raw": raw,
                        "parsed": Plan(task="t", steps=["s"]),
                        "parsing_error": None,
                    }

            return _Structured()

    off = _Capturing()
    build_planner(AgentModels(planner=off, executor=off))({"messages": []}, RunnableConfig())
    assert all(message.type != "system" for message in off.seen)

    on = _Capturing()
    build_planner(AgentModels(planner=on, executor=on), with_system_prompt=True)(
        {"messages": []}, RunnableConfig()
    )
    assert on.seen[0].type == "system"


def test_the_planner_never_sees_image_blocks() -> None:
    """Structured output needs no pixels; feeding them risks a provider 400."""
    from langchain_core.messages import HumanMessage
    from terminal_coding_agent.state import Plan

    class _CapturingPlanner:
        def __init__(self) -> None:
            self.seen: list = []

        def with_structured_output(self, schema, include_raw=False):  # noqa: ARG002
            outer = self

            class _Structured:
                def invoke(self, messages, config=None):  # noqa: ARG002
                    outer.seen = list(messages)
                    raw = AIMessage(
                        content="",
                        usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    )
                    return {"raw": raw, "parsed": Plan(task="t", steps=["s"]), "parsing_error": None}

            return _Structured()

    planner = _CapturingPlanner()
    node = build_planner(AgentModels(planner=planner, executor=planner))
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}

    node(
        {"messages": [HumanMessage(content=[{"type": "text", "text": "look"}, image])]},
        RunnableConfig(),
    )

    assert all(isinstance(message.content, str) for message in planner.seen)
    assert "look" in planner.seen[0].content
