import json
from contextlib import suppress
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
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

    with suppress(RuntimeError):
        node(_state(), RunnableConfig())

    trace = json.loads((tmp_path / ".agent" / "trace.json").read_text(encoding="utf-8"))
    assert trace["stop_reason"] == "completed"
    assert (tmp_path / ".agent" / "patch.diff").is_file()


def test_summary_relays_a_chat_reply_verbatim(tmp_path: Path) -> None:
    """Chat mode's last message IS the deliverable: no model call, no summarizing."""
    model = _StubModel(AIMessage(content="SUMMARY"))
    node = build_summary(AgentModels(planner=model, executor=model), worktree=tmp_path)
    state = _state(
        intent="chat",
        messages=[HumanMessage(content="写一篇作文"), AIMessage(content="the essay")],
    )

    updates = node(state, RunnableConfig())

    assert updates["messages"][-1].content == "the essay"
    assert updates["turns"] == 0, "no summary call may run"


def test_summary_summarizes_when_chat_produced_nothing(tmp_path: Path) -> None:
    """A failed chat node leaves the user's own message last; that is no reply to relay."""
    model = _StubModel(AIMessage(content="SUMMARY"))
    node = build_summary(AgentModels(planner=model, executor=model), worktree=tmp_path)
    state = _state(intent="chat", messages=[HumanMessage(content="写一篇作文")])

    updates = node(state, RunnableConfig())

    assert updates["messages"][-1].content == "SUMMARY"


def test_summary_summarizes_an_empty_chat_reply(tmp_path: Path) -> None:
    """Empty content is no reply: fall back to summarizing."""
    model = _StubModel(AIMessage(content="SUMMARY"))
    node = build_summary(AgentModels(planner=model, executor=model), worktree=tmp_path)
    state = _state(intent="chat", messages=[AIMessage(content="")])

    updates = node(state, RunnableConfig())

    assert updates["messages"][-1].content == "SUMMARY"


def test_summary_summarizes_a_work_task(tmp_path: Path) -> None:
    """Work mode never relays: the last step's reply is not the deliverable."""
    model = _StubModel(AIMessage(content="SUMMARY"))
    node = build_summary(AgentModels(planner=model, executor=model), worktree=tmp_path)
    state = _state(
        intent="work",
        messages=[HumanMessage(content="fix it"), AIMessage(content="I fixed it")],
    )

    updates = node(state, RunnableConfig())

    assert updates["messages"][-1].content == "SUMMARY"


def test_text_of_extracts_text_blocks_from_multimodal_content() -> None:
    from terminal_coding_agent.summary import _text_of

    message = HumanMessage(
        content=[
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
    )
    assert _text_of(message) == "what is this"


def test_chat_reply_survives_a_multimodal_user_message() -> None:
    from terminal_coding_agent.summary import _chat_reply

    messages = [
        HumanMessage(content=[{"type": "text", "text": "hi"}]),
        AIMessage(content="the answer"),
    ]
    assert _chat_reply(messages).content == "the answer"
