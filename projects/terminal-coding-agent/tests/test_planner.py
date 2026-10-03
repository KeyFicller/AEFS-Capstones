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


def test_make_plan_converts_model_crash_into_stop_reason() -> None:
    model = _PlannerStub(RuntimeError("400"))
    node = build_planner(AgentModels(planner=model, executor=model))

    out = node({"messages": []}, RunnableConfig())

    assert out["stop_reason"] == "planner_error:RuntimeError"
