import asyncio
from contextlib import suppress
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from terminal_coding_agent.checkpoint import build_checkpointer, checkpoint_path
from terminal_coding_agent.state import CodingAgentState, ToDoItem, ToDoStatus


def _tiny_graph() -> StateGraph:
    def node(state: CodingAgentState) -> dict:
        return {"todo_list": [ToDoItem(status=ToDoStatus.DONE, description="a")]}

    graph = StateGraph(CodingAgentState)
    graph.add_node("n", node)
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    return graph


def test_checkpoint_persists_across_saver_instances(tmp_path: Path) -> None:
    cfg = {"configurable": {"thread_id": "t1"}}
    first = _tiny_graph().compile(checkpointer=build_checkpointer(tmp_path))
    first.invoke({"messages": [HumanMessage(content="hi")]}, cfg)
    assert checkpoint_path(tmp_path).is_file()

    second = _tiny_graph().compile(checkpointer=build_checkpointer(tmp_path))
    values = second.get_state(cfg).values
    assert values["todo_list"][0].status is ToDoStatus.DONE


def test_ainvoke_checkpoints(tmp_path: Path) -> None:
    """Harbor's LangGraph runner only calls `ainvoke`; a sync-only saver raises there."""
    cfg = {"configurable": {"thread_id": "t-async"}}
    app = _tiny_graph().compile(checkpointer=build_checkpointer(tmp_path))
    values = asyncio.run(app.ainvoke({"messages": [HumanMessage(content="hi")]}, cfg))
    assert values["todo_list"][0].status is ToDoStatus.DONE
    assert checkpoint_path(tmp_path).is_file()


def test_run_resumes_from_interrupt(tmp_path: Path) -> None:
    cfg = {"configurable": {"thread_id": "t2"}}
    app = _tiny_graph().compile(checkpointer=build_checkpointer(tmp_path), interrupt_before=["n"])
    app.invoke({"messages": [HumanMessage(content="hi")]}, cfg)
    assert app.get_state(cfg).next == ("n",)

    resumed = app.invoke(None, cfg)
    assert resumed["todo_list"][0].status is ToDoStatus.DONE


def test_completed_nodes_are_not_rerun_on_resume(tmp_path: Path) -> None:
    """Resume granularity is the node: a committed node does not run again (spike #5)."""
    calls: list[str] = []

    def a(state: dict) -> dict:
        calls.append("a")
        return {"step": "a"}

    def b(state: dict) -> dict:
        calls.append("b")
        return {"step": "b"}

    graph = StateGraph(dict)
    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)

    cfg = {"configurable": {"thread_id": "t3"}}
    app = graph.compile(checkpointer=build_checkpointer(tmp_path), interrupt_after=["a"])
    app.invoke({}, cfg)
    assert calls == ["a"]

    app.invoke(None, cfg)
    assert calls == ["a", "b"]  # a did not rerun


def test_crashed_node_is_rerun_on_resume(tmp_path: Path) -> None:
    """A crash rolls back to before the node; resume re-enters it, so it runs twice (spike #6).

    Documents at-least-once semantics: side effects in a crashed node are NOT undone.
    """
    calls: list[str] = []
    fail = {"on": True}

    def a(state: dict) -> dict:
        calls.append("a")
        return {"step": "a"}

    def b(state: dict) -> dict:
        calls.append("b")
        if fail["on"]:
            raise RuntimeError("model 400")
        return {"step": "b"}

    graph = StateGraph(dict)
    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)

    cfg = {"configurable": {"thread_id": "t4"}}
    app = graph.compile(checkpointer=build_checkpointer(tmp_path))

    with suppress(RuntimeError):
        app.invoke({}, cfg)
    assert app.get_state(cfg).next == ("b",)

    fail["on"] = False
    app.invoke(None, cfg)
    assert calls.count("b") == 2  # once before the crash, once after resume
