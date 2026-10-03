import os
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from terminal_coding_agent import graph as graph_module
from terminal_coding_agent.graph import (
    _after_end_task,
    _after_make_plan,
    _after_recover,
    _after_recover_gated,
    make_graph,
)
from terminal_coding_agent.models import AgentModels
from terminal_coding_agent.recover import build_recover
from terminal_coding_agent.state import Plan, ToDoItem, ToDoStatus


def test_mermaid_png_is_skipped_without_a_path(tmp_path: Path) -> None:
    os.environ["DEEPSEEK_API_KEY"] = "test"
    calls: list[object] = []

    def fake_png(self, **kwargs):
        calls.append(self)
        return b"\x89PNG\r\n\x1a\nfake"

    with patch("langchain_core.runnables.graph.Graph.draw_mermaid_png", fake_png):
        make_graph({"configurable": {"worktree": tmp_path, "show_mermaid": True}})
    os.environ.pop("DEEPSEEK_API_KEY")

    assert calls == []


def test_mermaid_path_writes_png(tmp_path: Path, capsys) -> None:
    os.environ["DEEPSEEK_API_KEY"] = "test"
    png_path = tmp_path / "diagrams" / "agent.png"

    def fake_png(self, **kwargs):
        return b"\x89PNG\r\n\x1a\nfake"

    with patch("langchain_core.runnables.graph.Graph.draw_mermaid_png", fake_png):
        make_graph({"configurable": {"worktree": tmp_path, "mermaid_path": png_path}})
    os.environ.pop("DEEPSEEK_API_KEY")

    assert "flowchart" not in capsys.readouterr().out
    assert png_path.read_bytes().startswith(b"\x89PNG")


def test_make_graph(tmp_path: Path) -> None:
    # Just for testing
    os.environ["DEEPSEEK_API_KEY"] = "test"
    agent = make_graph(
        {
            "configurable": {
                "worktree": tmp_path,
            },
        }
    )
    os.environ.pop("DEEPSEEK_API_KEY")

    assert agent is not None


def _state(todo_list, **extra):
    state = {
        "messages": [],
        "todo_list": todo_list,
        "turns": 0,
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cost_rmb": 0.0,
        "stop_reason": None,
        "current_task_index": None,
        "replan_count": 0,
        "blocked_reason": None,
        "blocked_evidence": [],
    }
    state.update(extra)
    return state


def test_end_task_routes_blocked_to_recover() -> None:
    state = _state(
        [
            ToDoItem(status=ToDoStatus.DONE, description="a"),
            ToDoItem(status=ToDoStatus.FAILED, description="b"),
        ],
        blocked_reason="pytest fails",
    )

    assert _after_end_task(state) == "recover"


def test_end_task_prefers_stop_reason_over_block() -> None:
    state = _state(
        [ToDoItem(status=ToDoStatus.FAILED, description="b")],
        blocked_reason="pytest fails",
        stop_reason="max_turns",
    )

    assert _after_end_task(state) == "summary"


def test_end_task_finishes_when_only_done_and_deprecated_remain() -> None:
    state = _state(
        [
            ToDoItem(status=ToDoStatus.DEPRECATED, description="a"),
            ToDoItem(status=ToDoStatus.DONE, description="b"),
        ]
    )

    assert _after_end_task(state) == "summary"


def test_end_task_continues_with_pending_work() -> None:
    state = _state(
        [
            ToDoItem(status=ToDoStatus.DONE, description="a"),
            ToDoItem(status=ToDoStatus.PENDING, description="b"),
        ]
    )

    assert _after_end_task(state) == "start_task"


def test_recover_resumes_or_ends() -> None:
    assert _after_recover(_state([ToDoItem(status=ToDoStatus.PENDING, description="a")])) == (
        "start_task"
    )
    assert _after_recover(_state([], stop_reason="recover_exhausted")) == "summary"


def test_recover_routes_to_summary_when_the_replan_leaves_nothing_pending(
    tmp_path: Path,
) -> None:
    """An empty replan retires the failed step, so the run must end, not re-enter start_task."""

    def empty_replan(state, config):  # noqa: ARG001 - mirrors make_plan
        return {"todo_list": [], "messages": []}

    recover = build_recover(empty_replan, worktree=tmp_path)
    state = _state([ToDoItem(status=ToDoStatus.FAILED, description="b")])

    planned = recover(state, {})

    # start_task asserts on a missing PENDING item, so this route must never be taken.
    assert _after_recover({**state, **planned}) == "summary"


def test_settled_only_plans_end_the_run_from_either_router() -> None:
    """Both routers share one rule: no pending work left means summary."""
    settled_only = (
        [ToDoItem(status=ToDoStatus.DONE, description="a")],
        [ToDoItem(status=ToDoStatus.DEPRECATED, description="a")],
        [
            ToDoItem(status=ToDoStatus.DONE, description="a"),
            ToDoItem(status=ToDoStatus.DEPRECATED, description="b"),
        ],
    )

    for todos in settled_only:
        assert _after_end_task(_state(todos)) == "summary"
        assert _after_recover(_state(todos)) == "summary"


def test_make_plan_ends_when_plan_is_empty() -> None:
    assert _after_make_plan(_state([])) == "summary"
    assert (
        _after_make_plan(_state([ToDoItem(status=ToDoStatus.PENDING, description="a")]))
        == "start_task"
    )


def test_compiled_graph_uses_the_module_routers(tmp_path: Path) -> None:
    """The extracted routers must be the ones wired, not a leftover closure."""
    os.environ["DEEPSEEK_API_KEY"] = "test"
    agent = make_graph({"configurable": {"worktree": tmp_path}})
    os.environ.pop("DEEPSEEK_API_KEY")

    wired = {node: list(branches) for node, branches in agent.builder.branches.items()}
    assert wired["make_plan"] == ["_after_make_plan"]
    assert wired["start_task"] == ["_after_start"]
    assert wired["end_task"] == ["_after_end_task"]
    assert wired["recover"] == ["_after_recover"]


def test_make_graph_persists_checkpoints(tmp_path: Path) -> None:
    os.environ["DEEPSEEK_API_KEY"] = "test"
    agent = make_graph({"configurable": {"worktree": tmp_path}})
    os.environ.pop("DEEPSEEK_API_KEY")

    cfg = {"configurable": {"worktree": tmp_path, "thread_id": "t1"}}
    assert agent.get_state(cfg) is not None
    assert (tmp_path / ".agent" / "checkpoints.sqlite").is_file()


def test_invoke_without_thread_id_uses_injected_default(tmp_path: Path) -> None:
    os.environ["DEEPSEEK_API_KEY"] = "test"
    agent = make_graph({"configurable": {"worktree": tmp_path}})
    os.environ.pop("DEEPSEEK_API_KEY")

    # A checked graph raises without thread_id; with_config must supply a default.
    assert agent.config["configurable"]["thread_id"]


def test_compiled_make_plan_announces_the_plan(tmp_path: Path, monkeypatch, capsys) -> None:
    """The plan must reach the console during a real run, not only in the demo."""
    plan = Plan(task="fix the typo", steps=["locate the typo", "edit it"])

    class _Structured:
        def invoke(self, messages, config=None):  # noqa: A002 - mirrors Runnable
            return {"raw": AIMessage(content=""), "parsed": plan}

    class _Planner:
        def with_structured_output(self, schema, include_raw=False):
            return _Structured()

    stub = AgentModels(planner=_Planner(), executor=_Planner())
    monkeypatch.setattr(graph_module, "build_models", lambda config: stub)
    agent = make_graph({"configurable": {"worktree": tmp_path}})

    agent.builder.nodes["make_plan"].runnable.invoke(_state([]), None)

    out = capsys.readouterr().out
    assert "[-]  locate the typo" in out
    assert "[-]  edit it" in out


def _spy_on_execute_nodes(monkeypatch) -> dict:
    captured: dict = {}
    real_build = graph_module.build_execute_nodes

    def spy(models, tools, **kwargs):
        captured.update(kwargs)
        return real_build(models, tools, **kwargs)

    monkeypatch.setattr(graph_module, "build_execute_nodes", spy)
    return captured


def test_make_graph_forwards_the_injected_tool_renderer(tmp_path: Path, monkeypatch) -> None:
    """The renderer travels config -> graph -> execute nodes, like todo_renderer."""
    os.environ["DEEPSEEK_API_KEY"] = "test"
    captured = _spy_on_execute_nodes(monkeypatch)
    renderer = lambda *_: None

    make_graph({"configurable": {"worktree": tmp_path, "tool_renderer": renderer}})
    os.environ.pop("DEEPSEEK_API_KEY")

    assert captured["tool_renderer"] is renderer


def test_make_graph_ignores_a_non_callable_tool_renderer(tmp_path: Path, monkeypatch) -> None:
    """A bad renderer in config must degrade to 'no tool log', not crash the graph."""
    os.environ["DEEPSEEK_API_KEY"] = "test"
    captured = _spy_on_execute_nodes(monkeypatch)

    make_graph({"configurable": {"worktree": tmp_path, "tool_renderer": "not callable"}})
    os.environ.pop("DEEPSEEK_API_KEY")

    assert captured["tool_renderer"] is None


def test_make_graph_records_a_sequence_only_when_a_path_is_configured(
    tmp_path: Path, monkeypatch
) -> None:
    """`sequence_path` is the only switch; without it the recorder must stay off."""
    os.environ["DEEPSEEK_API_KEY"] = "test"
    captured = _spy_on_execute_nodes(monkeypatch)

    make_graph({"configurable": {"worktree": tmp_path}})
    assert captured["sequence_events"] is None

    make_graph(
        {
            "configurable": {
                "worktree": tmp_path,
                "sequence_path": str(tmp_path / "sequence.png"),
            }
        }
    )
    os.environ.pop("DEEPSEEK_API_KEY")

    assert captured["sequence_events"] == []


def test_gated_graph_adds_the_approval_node(tmp_path: Path) -> None:
    os.environ["DEEPSEEK_API_KEY"] = "test"
    agent = make_graph({"configurable": {"worktree": tmp_path, "enable_hitl": True}})
    os.environ.pop("DEEPSEEK_API_KEY")

    assert "await_plan_approval" in agent.builder.nodes
    assert list(agent.builder.branches["make_plan"]) == ["_after_make_plan_gated"]


def test_ungated_graph_has_no_approval_node(tmp_path: Path) -> None:
    """Harbor parity: without the flag the node set and routers are unchanged."""
    os.environ["DEEPSEEK_API_KEY"] = "test"
    agent = make_graph({"configurable": {"worktree": tmp_path}})
    os.environ.pop("DEEPSEEK_API_KEY")

    assert "await_plan_approval" not in agent.builder.nodes
    assert list(agent.builder.branches["make_plan"]) == ["_after_make_plan"]


def test_make_graph_forwards_enable_hitl_to_the_execute_nodes(tmp_path: Path, monkeypatch) -> None:
    """Harbor parity: absent by default, so the middleware table cannot change."""
    os.environ["DEEPSEEK_API_KEY"] = "test"
    captured = _spy_on_execute_nodes(monkeypatch)

    make_graph({"configurable": {"worktree": tmp_path}})
    assert captured["enable_hitl"] is False

    make_graph({"configurable": {"worktree": tmp_path, "enable_hitl": True}})
    os.environ.pop("DEEPSEEK_API_KEY")

    assert captured["enable_hitl"] is True


def test_gated_graph_gates_the_replan_too(tmp_path: Path) -> None:
    """A replan is entered from `recover`, not `make_plan` — the gate needs its own edge."""
    os.environ["DEEPSEEK_API_KEY"] = "test"
    agent = make_graph({"configurable": {"worktree": tmp_path, "enable_hitl": True}})
    os.environ.pop("DEEPSEEK_API_KEY")

    assert list(agent.builder.branches["recover"]) == ["_after_recover_gated"]


def test_after_recover_gated_only_diverts_a_live_replan() -> None:
    """Mirrors `_after_recover` except for the one case that must reach the gate."""
    pending = [ToDoItem(status=ToDoStatus.PENDING, description="a")]

    assert _after_recover_gated(_state(pending)) == "await_plan_approval"
    assert _after_recover_gated(_state([], stop_reason="recover_exhausted")) == "summary"
    assert (
        _after_recover_gated(_state([ToDoItem(status=ToDoStatus.DONE, description="a")]))
        == "summary"
    )
