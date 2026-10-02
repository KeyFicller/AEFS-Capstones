import json
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from terminal_coding_agent.models import AgentModels
from terminal_coding_agent.state import ToDoItem, ToDoStatus
from terminal_coding_agent.summary import build_summary


class _StubModel:
    def __init__(self, reply: AIMessage) -> None:
        self.reply = reply

    def invoke(self, input, config=None):  # noqa: A002 - mirrors BaseChatModel
        return self.reply


def _state(**extra) -> dict:
    state = {
        "messages": [],
        "todo_list": [],
        "turns": 0,
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cost_rmb": 0.0,
        "stop_reason": None,
        "current_task_index": None,
    }
    state.update(extra)
    return state


def test_summary_writes_the_stop_trace(tmp_path: Path) -> None:
    model = _StubModel(AIMessage(content="done"))
    node = build_summary(AgentModels(planner=model, executor=model), worktree=tmp_path)
    state = _state(
        todo_list=[ToDoItem(status=ToDoStatus.DONE, description="a")],
        stop_reason="max_turns",
    )

    node(state, RunnableConfig())

    trace = json.loads((tmp_path / ".agent" / "trace.json").read_text(encoding="utf-8"))
    assert trace["stop_reason"] == "max_turns"
    assert trace["todo_list"][0]["status"] == "DONE"
    assert (tmp_path / ".agent" / "patch.diff").is_file()


def test_summary_writes_trace_even_if_the_model_fails(tmp_path: Path) -> None:
    class _Boom(_StubModel):
        def invoke(self, input, config=None):  # noqa: A002
            raise RuntimeError("model down")

    model = _Boom(AIMessage(content="unused"))
    node = build_summary(AgentModels(planner=model, executor=model), worktree=tmp_path)

    try:
        node(_state(), RunnableConfig())
    except RuntimeError:
        pass

    trace = json.loads((tmp_path / ".agent" / "trace.json").read_text(encoding="utf-8"))
    assert trace["stop_reason"] == "completed"
    assert (tmp_path / ".agent" / "patch.diff").is_file()
