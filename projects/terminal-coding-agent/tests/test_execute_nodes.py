"""Mocked execute-node wiring: status markers, budget hard-stop, final-message filter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from terminal_coding_agent.executor import _final_agent_message, build_execute_nodes
from terminal_coding_agent.state import ToDoItem, ToDoStatus


def _base_state(todos: list[ToDoItem], **extra):
    state = {
        "messages": [],
        "todo_list": todos,
        "turns": 0,
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cost_usd": 0.0,
        "stop_reason": None,
        "current_task_index": None,
    }
    state.update(extra)
    return state


def test_start_then_end_marks_in_progress_then_done(tmp_path: Path) -> None:
    nodes = build_execute_nodes(MagicMock(), tools=[], worktree=tmp_path)
    state = _base_state([ToDoItem(status=ToDoStatus.PENDING, description="fix typo")])

    started = nodes["start_task"](state)
    assert started["todo_list"][0].status == ToDoStatus.IN_PROGRESS
    assert started["current_task_index"] == 0

    ended = nodes["end_task"]({**state, **started})
    assert ended["todo_list"][0].status == ToDoStatus.DONE
    assert ended["current_task_index"] is None
    assert (tmp_path / ".agent" / "trace.json").is_file()


def test_start_task_hard_stops_when_turns_exhausted(tmp_path: Path) -> None:
    nodes = build_execute_nodes(MagicMock(), tools=[], worktree=tmp_path)
    state = _base_state(
        [ToDoItem(status=ToDoStatus.PENDING, description="never runs")],
        turns=50,
    )

    out = nodes["start_task"](state)
    assert out["stop_reason"] == "max_turns"
    assert "todo_list" not in out or out.get("current_task_index") is None

    trace = json.loads((tmp_path / ".agent" / "trace.json").read_text(encoding="utf-8"))
    assert trace["stop_reason"] == "max_turns"


def test_final_agent_message_keeps_last_plain_ai() -> None:
    messages = [
        HumanMessage(content="do it"),
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "x"}, "id": "1"}],
        ),
        ToolMessage(content="file", tool_call_id="1"),
        AIMessage(content="fixed the typo"),
    ]
    final = _final_agent_message(messages)
    assert final is not None
    assert final.content == "fixed the typo"
