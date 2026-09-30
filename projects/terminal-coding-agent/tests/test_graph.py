from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from terminal_coding_agent.graph import (
    _after_end_task,
    _after_make_plan,
    _after_recover,
    make_graph,
)
from terminal_coding_agent.state import ToDoItem, ToDoStatus


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
    assert _after_recover(_state([])) == "start_task"
    assert _after_recover(_state([], stop_reason="recover_exhausted")) == "summary"


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
