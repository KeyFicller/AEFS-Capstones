import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from terminal_coding_agent import graph as graph_module
from terminal_coding_agent.graph import make_graph
from terminal_coding_agent.models import AgentModels
from terminal_coding_agent.session import (
    pending_interrupts,
    reset_updates,
    resume_turn,
    run_task_turn,
)
from terminal_coding_agent.state import CodingAgentState


def test_reset_covers_every_state_field_except_messages() -> None:
    """A new state field must force a decision here, not silently survive a turn."""
    assert set(reset_updates()) == set(CodingAgentState.__annotations__) - {"messages"}


def test_reset_zeroes_the_ledger_and_the_plan() -> None:
    reset = reset_updates()

    assert reset["turns"] == 0
    assert reset["tokens"] == 0
    assert reset["input_tokens"] == 0
    assert reset["output_tokens"] == 0
    assert reset["cache_read_tokens"] == 0
    assert reset["cost_rmb"] == 0.0
    assert reset["todo_list"] == []
    assert reset["stop_reason"] is None
    assert reset["current_task_index"] is None
    assert reset["replan_count"] == 0
    assert reset["blocked_reason"] is None
    assert reset["blocked_evidence"] == []


def test_reset_returns_fresh_mutables() -> None:
    """Two turns must not share a list, or one turn's edits leak into the next."""
    first = reset_updates()
    first["todo_list"].append("leak")
    first["blocked_evidence"].append({"leak": True})

    second = reset_updates()

    assert second["todo_list"] == []
    assert second["blocked_evidence"] == []


class _OrderGraph:
    """Records the order of reset vs. invoke so the reset cannot silently drift."""

    def __init__(self, result: dict | None = None) -> None:
        self.log: list[str] = []
        self.payloads: list[dict] = []
        self.result = result or {}

    def update_state(self, config, values) -> None:  # noqa: ARG002
        self.log.append("reset")

    def invoke(self, payload, config) -> dict:  # noqa: ARG002
        self.log.append("invoke")
        self.payloads.append(payload)
        return self.result


def test_run_task_turn_resets_before_invoking() -> None:
    graph = _OrderGraph({"turns": 4})

    run_task_turn(graph=graph, config={}, message=HumanMessage(content="fix the typo"))

    assert graph.log == ["reset", "invoke"]


def test_run_task_turn_sends_the_text_as_a_human_message() -> None:
    graph = _OrderGraph()

    run_task_turn(graph=graph, config={}, message=HumanMessage(content="fix the typo"))

    sent = graph.payloads[0]["messages"]
    assert len(sent) == 1
    assert sent[0].content == "fix the typo"


def test_run_task_turn_returns_the_graph_state() -> None:
    graph = _OrderGraph({"turns": 7, "stop_reason": "max_turns"})

    result = run_task_turn(graph=graph, config={}, message=HumanMessage(content="t"))

    assert result["stop_reason"] == "max_turns"


def _stub_models() -> AgentModels:
    class _Planner:
        def with_structured_output(self, schema, include_raw: bool = False):  # noqa: ARG002
            class _Structured:
                def invoke(self, messages, config=None):  # noqa: ARG002
                    raise AssertionError("planner must not run in this test")

            return _Structured()

        def invoke(self, messages, config=None):  # noqa: ARG002
            return AIMessage(content="unused")

    stub = _Planner()
    return AgentModels(planner=stub, executor=stub)


def test_real_graph_accepts_a_reset_on_a_fresh_thread(tmp_path: Path, monkeypatch) -> None:
    """The reset must work against the real compiled graph + SqliteSaver, not just a stub."""
    os.environ["DEEPSEEK_API_KEY"] = "test"
    monkeypatch.setattr(graph_module, "build_models", lambda config: _stub_models())
    agent = make_graph({"configurable": {"worktree": tmp_path}})
    os.environ.pop("DEEPSEEK_API_KEY")

    config = {"configurable": {"worktree": tmp_path, "thread_id": "t-reset"}}
    agent.update_state(config, reset_updates())

    values = agent.get_state(config).values
    assert values["turns"] == 0
    assert values["todo_list"] == []


def test_resume_turn_sends_a_resume_command_without_resetting() -> None:
    graph = _OrderGraph({"stop_reason": "completed"})

    result = resume_turn(graph=graph, config={}, decision="approve")

    assert graph.log == ["invoke"]  # NO reset: the pause *is* the checkpoint
    assert isinstance(graph.payloads[0], Command)
    assert result["stop_reason"] == "completed"


def test_pending_interrupts_reads_the_paused_snapshot() -> None:
    class _Task:
        interrupts = ("a", "b")

    class _Snap:
        tasks = (_Task(),)

    class _Graph:
        def get_state(self, config):
            return _Snap()

    assert pending_interrupts(_Graph(), {}) == ["a", "b"]


def test_pending_interrupts_is_empty_when_the_graph_is_not_paused() -> None:
    class _Snap:
        tasks = ()

    class _Graph:
        def get_state(self, config):
            return _Snap()

    assert pending_interrupts(_Graph(), {}) == []


def test_reset_clears_the_answer_mode() -> None:
    """A stale 'answer' from a previous turn must not leak into the next turn."""
    assert reset_updates()["mode"] is None
