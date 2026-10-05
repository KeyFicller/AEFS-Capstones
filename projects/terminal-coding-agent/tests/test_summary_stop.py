import json
from contextlib import suppress
from pathlib import Path

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
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


class _StreamModel:
    """Streams the given chunks; `invoke` must not run when the stream produced any."""

    def __init__(self, chunks: list[AIMessageChunk]) -> None:
        self.chunks = chunks

    def stream(self, input, config=None, **kwargs):  # noqa: A002 - mirrors BaseChatModel
        yield from self.chunks

    def invoke(self, input, config=None):  # noqa: A002
        raise AssertionError("stream produced chunks; invoke must not run")


def test_summary_streams_the_running_text_into_the_renderer(tmp_path: Path) -> None:
    """The renderer gets the whole text so far after each chunk, not the delta."""
    chunks = [
        AIMessageChunk(
            content="he",
            usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
        ),
        AIMessageChunk(
            content="llo",
            usage_metadata={"input_tokens": 0, "output_tokens": 2, "total_tokens": 2},
        ),
    ]
    model = _StreamModel(chunks)
    seen: list[str] = []
    node = build_summary(
        AgentModels(planner=model, executor=model),
        worktree=tmp_path,
        summary_renderer=seen.append,
    )

    updates = node(_state(), RunnableConfig())

    assert seen == ["he", "hello"], "each chunk must show the summary so far"
    reply = updates["messages"][-1]
    assert isinstance(reply, AIMessage)
    assert reply.content == "hello"
    assert reply.usage_metadata == {"input_tokens": 3, "output_tokens": 3, "total_tokens": 6}
    assert updates["turns"] == 1, "the streamed call must still be budgeted"
    trace = json.loads((tmp_path / ".agent" / "trace.json").read_text(encoding="utf-8"))
    assert trace["turns"] == 1


def test_summary_falls_back_to_one_shot_when_the_stream_is_empty(tmp_path: Path) -> None:
    """A model that streams nothing must not swallow the reply: the panel still appears."""
    class _Empty(_StreamModel):
        def stream(self, input, config=None, **kwargs):  # noqa: A002
            return iter(())

        def invoke(self, input, config=None):  # noqa: A002
            return AIMessage(content="fallback")

    model = _Empty([])
    seen: list[str] = []
    node = build_summary(
        AgentModels(planner=model, executor=model),
        worktree=tmp_path,
        summary_renderer=seen.append,
    )

    updates = node(_state(), RunnableConfig())

    assert seen == ["fallback"]
    assert updates["messages"][-1].content == "fallback"


def test_summary_relays_the_answer_verbatim(tmp_path: Path) -> None:
    """Answer mode's last message IS the deliverable: no model call, no summarizing."""
    model = _StubModel(AIMessage(content="SUMMARY"))
    node = build_summary(AgentModels(planner=model, executor=model), worktree=tmp_path)
    state = _state(
        mode="answer",
        messages=[HumanMessage(content="写一篇作文"), AIMessage(content="the essay")],
    )

    updates = node(state, RunnableConfig())

    assert updates["messages"][-1].content == "the essay"
    assert updates["turns"] == 0, "no summary call may run"


def test_summary_renders_the_answer(tmp_path: Path) -> None:
    """The panel must show the answer too, not only the returned message."""
    model = _StubModel(AIMessage(content="SUMMARY"))
    seen: list[str] = []
    node = build_summary(
        AgentModels(planner=model, executor=model),
        worktree=tmp_path,
        summary_renderer=seen.append,
    )
    state = _state(mode="answer", messages=[AIMessage(content="the essay")])

    node(state, RunnableConfig())

    assert seen == ["the essay"]


def test_summary_summarizes_when_answer_mode_produced_nothing(tmp_path: Path) -> None:
    """A failed answer node leaves the user's own message last; that is no reply to relay."""
    model = _StubModel(AIMessage(content="SUMMARY"))
    node = build_summary(AgentModels(planner=model, executor=model), worktree=tmp_path)
    state = _state(mode="answer", messages=[HumanMessage(content="写一篇作文")])

    updates = node(state, RunnableConfig())

    assert updates["messages"][-1].content == "SUMMARY"


def test_summary_summarizes_an_empty_answer(tmp_path: Path) -> None:
    """Empty content is no answer: fall back to summarizing."""
    model = _StubModel(AIMessage(content="SUMMARY"))
    node = build_summary(AgentModels(planner=model, executor=model), worktree=tmp_path)
    state = _state(mode="answer", messages=[AIMessage(content="")])

    updates = node(state, RunnableConfig())

    assert updates["messages"][-1].content == "SUMMARY"


def test_summary_summarizes_a_work_task(tmp_path: Path) -> None:
    """Work mode never relays: the last step's reply is not the deliverable."""
    model = _StubModel(AIMessage(content="SUMMARY"))
    node = build_summary(AgentModels(planner=model, executor=model), worktree=tmp_path)
    state = _state(
        mode="work",
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


def test_answer_reply_survives_a_multimodal_user_message() -> None:
    from terminal_coding_agent.summary import _answer_reply

    messages = [
        HumanMessage(content=[{"type": "text", "text": "hi"}]),
        AIMessage(content="the answer"),
    ]
    assert _answer_reply(messages).content == "the answer"
