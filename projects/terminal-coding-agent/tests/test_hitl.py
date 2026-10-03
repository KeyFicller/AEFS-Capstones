from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from terminal_coding_agent.checkpoint import build_checkpointer
from terminal_coding_agent.graph import _after_plan_approval, _await_plan_approval
from terminal_coding_agent.state import CodingAgentState, ToDoItem, ToDoStatus


def _seed(state: CodingAgentState) -> dict:
    return {"todo_list": [ToDoItem(ToDoStatus.PENDING, "step-a")]}


def _ran(state: CodingAgentState) -> dict:
    return {"todo_list": [*state["todo_list"], ToDoItem(ToDoStatus.DONE, "ran")]}


def _summary(state: CodingAgentState) -> dict:
    return {"stop_reason": state.get("stop_reason") or "completed"}


def _gate_graph(tmp_path: Path, *, seed: Callable[..., dict] = _seed):
    graph = StateGraph(CodingAgentState)
    graph.add_node("seed", seed)
    graph.add_node("gate", _await_plan_approval)
    graph.add_node("ran", _ran)
    graph.add_node("summary", _summary)
    graph.add_edge(START, "seed")
    graph.add_edge("seed", "gate")
    graph.add_conditional_edges(
        "gate", _after_plan_approval, {"start_task": "ran", "summary": "summary"}
    )
    graph.add_edge("ran", "summary")
    graph.add_edge("summary", END)
    return graph.compile(checkpointer=build_checkpointer(tmp_path))


def _cfg(name: str) -> dict:
    return {"configurable": {"thread_id": name}}


def test_gate_pauses_before_the_agent_runs(tmp_path: Path) -> None:
    app = _gate_graph(tmp_path)
    out = app.invoke({"messages": []}, _cfg("t1"))

    assert "step-a" in out["__interrupt__"][0].value["plan"]
    assert app.get_state(_cfg("t1")).next == ("gate",)


def test_approve_reaches_the_agent(tmp_path: Path) -> None:
    app = _gate_graph(tmp_path)
    app.invoke({"messages": []}, _cfg("t2"))

    out = app.invoke(Command(resume="approve"), _cfg("t2"))

    assert out["stop_reason"] == "completed"
    assert any(item.description == "ran" for item in out["todo_list"])


def test_reject_stops_without_running_the_agent(tmp_path: Path) -> None:
    app = _gate_graph(tmp_path)
    app.invoke({"messages": []}, _cfg("t3"))

    out = app.invoke(Command(resume="reject"), _cfg("t3"))

    assert out["stop_reason"] == "plan_rejected"
    assert all(item.description != "ran" for item in out.get("todo_list") or [])


def test_pending_interrupt_survives_a_new_graph_instance(tmp_path: Path) -> None:
    first = _gate_graph(tmp_path)
    first.invoke({"messages": []}, _cfg("t4"))

    second = _gate_graph(tmp_path)  # fresh compile, same sqlite file
    snapshot = second.get_state(_cfg("t4"))
    pending = [i for task in snapshot.tasks for i in task.interrupts]
    assert pending  # readable without the original process

    out = second.invoke(Command(resume="approve"), _cfg("t4"))
    assert out["stop_reason"] == "completed"


def test_the_interrupt_node_reruns_from_the_top_on_resume(tmp_path: Path) -> None:
    """At-least-once: code *before* interrupt() runs again on resume."""
    entries: list[int] = []

    def gate(state: CodingAgentState) -> dict:
        entries.append(1)
        return {} if interrupt({"plan": "x"}) == "approve" else {"stop_reason": "plan_rejected"}

    graph = StateGraph(CodingAgentState)
    graph.add_node("gate", gate)
    graph.add_node("summary", _summary)
    graph.add_edge(START, "gate")
    graph.add_edge("gate", "summary")
    graph.add_edge("summary", END)
    app = graph.compile(checkpointer=build_checkpointer(tmp_path))

    app.invoke({"messages": []}, _cfg("t5"))
    app.invoke(Command(resume="approve"), _cfg("t5"))

    assert len(entries) == 2


def _replanned(state: CodingAgentState) -> dict:
    """What `recover` leaves behind: a fresh plan with one replan already counted."""
    return {
        "todo_list": [ToDoItem(ToDoStatus.PENDING, "step-a")],
        "replan_count": 1,
    }


def test_reject_after_a_replan_is_labelled_replan_rejected(tmp_path: Path) -> None:
    """The initial plan and a mid-task replan are different failure modes; keep them apart."""
    app = _gate_graph(tmp_path, seed=_replanned)
    app.invoke({"messages": []}, _cfg("t6"))

    out = app.invoke(Command(resume="reject"), _cfg("t6"))

    assert out["stop_reason"] == "replan_rejected"
    assert all(item.description != "ran" for item in out.get("todo_list") or [])
