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
        "cost_rmb": 0.0,
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

def test_run_agent_blocked_records_evidence_without_transcript(tmp_path, monkeypatch) -> None:
    from terminal_coding_agent import executor as executor_module

    blocked_messages = [
        HumanMessage(content="fix the import"),
        AIMessage(
            content="",
            tool_calls=[{"name": "report_blocked", "args": {"reason": "pytest fails"}, "id": "1"}],
        ),
        ToolMessage(content="Blocked reported.", tool_call_id="1", name="report_blocked"),
        AIMessage(content="Blocked: pytest fails"),
    ]

    class FakeAgent:
        def invoke(self, payload, config):
            return {"messages": blocked_messages, "blocked_reason": "pytest fails"}

    monkeypatch.setattr(executor_module, "create_agent", lambda **kwargs: FakeAgent())
    nodes = executor_module.build_execute_nodes(MagicMock(), tools=[], worktree=tmp_path)
    state = _base_state(
        [ToDoItem(status=ToDoStatus.IN_PROGRESS, description="fix the import")],
        current_task_index=0,
    )

    out = nodes["run_agent"](state, {})

    assert out["blocked_reason"] == "pytest fails"
    assert [entry["action"] for entry in out["blocked_evidence"]] == [
        "report_blocked",
        "report_blocked",
    ]
    assert "messages" not in out

    ended = nodes["end_task"]({**state, **out})

    assert ended["todo_list"][0].status == ToDoStatus.FAILED
    assert ended["current_task_index"] is None

    trace = json.loads((tmp_path / ".agent" / "trace.json").read_text(encoding="utf-8"))
    assert trace["stop_reason"] == "blocked"
    assert trace["todo_list"][0]["status"] == "FAILED"


def test_start_task_clears_a_stale_block_report(tmp_path: Path) -> None:
    nodes = build_execute_nodes(MagicMock(), tools=[], worktree=tmp_path)
    state = _base_state(
        [ToDoItem(status=ToDoStatus.PENDING, description="next task")],
        blocked_reason="boom",
        blocked_evidence=[{"action": "run_shell", "detail": "x"}],
    )

    out = nodes["start_task"](state)

    assert out["blocked_reason"] is None
    assert out["blocked_evidence"] == []


def test_success_after_a_block_is_not_marked_failed(tmp_path: Path, monkeypatch) -> None:
    from terminal_coding_agent import executor as executor_module

    class FakeAgent:
        def invoke(self, payload, config):
            return {"messages": [AIMessage(content="done it")]}

    monkeypatch.setattr(executor_module, "create_agent", lambda **kwargs: FakeAgent())
    nodes = executor_module.build_execute_nodes(MagicMock(), tools=[], worktree=tmp_path)
    state = _base_state(
        [
            ToDoItem(status=ToDoStatus.FAILED, description="blocked task"),
            ToDoItem(status=ToDoStatus.PENDING, description="innocent task"),
        ],
        blocked_reason="boom",
        blocked_evidence=[{"action": "run_shell", "detail": "x"}],
    )

    started = nodes["start_task"](state)
    ran = nodes["run_agent"]({**state, **started}, {})
    ended = nodes["end_task"]({**state, **started, **ran})

    assert ran.get("blocked_reason") is None
    assert ended["todo_list"][1].status == ToDoStatus.DONE


def test_end_task_marks_done_when_not_blocked(tmp_path, monkeypatch) -> None:
    from terminal_coding_agent import executor as executor_module

    class FakeAgent:
        def invoke(self, payload, config):
            return {"messages": [AIMessage(content="fixed")]}

    monkeypatch.setattr(executor_module, "create_agent", lambda **kwargs: FakeAgent())
    nodes = executor_module.build_execute_nodes(MagicMock(), tools=[], worktree=tmp_path)
    state = _base_state(
        [ToDoItem(status=ToDoStatus.IN_PROGRESS, description="fix typo")],
        current_task_index=0,
    )

    out = nodes["run_agent"](state, {})

    assert out.get("blocked_reason") is None
    assert out["messages"][0].content == "fixed"

    ended = nodes["end_task"]({**state, **out})

    assert ended["todo_list"][0].status == ToDoStatus.DONE

