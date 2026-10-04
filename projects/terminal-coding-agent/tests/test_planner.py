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


def test_answer_mode_plans_without_todos() -> None:
    from terminal_coding_agent.state import RoutingPlan

    parsed = RoutingPlan(task="写一篇作文", mode="answer", steps=["写一篇作文"])
    planner = _SchemaStub(parsed)
    node = build_planner(AgentModels(planner=planner, executor=planner), enable_answer=True)

    out = node({"messages": []}, RunnableConfig())

    assert planner.schema is RoutingPlan
    assert out["mode"] == "answer"
    assert "todo_list" not in out
    assert "messages" not in out, "answer mode must not write a plan message"


def test_work_plan_keeps_the_plain_schema_when_the_flag_is_off() -> None:
    from terminal_coding_agent.state import Plan

    parsed = Plan(task="fix the typo", steps=["locate it", "edit it"])
    planner = _SchemaStub(parsed)
    node = build_planner(AgentModels(planner=planner, executor=planner))

    out = node({"messages": []}, RunnableConfig())

    assert planner.schema is Plan, "Harbor parity: no mode field without the flag"
    assert out["mode"] == "work"
    assert [item.description for item in out["todo_list"]] == ["locate it", "edit it"]
