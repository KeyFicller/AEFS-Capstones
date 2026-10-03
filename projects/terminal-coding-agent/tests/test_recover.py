import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from terminal_coding_agent.recover import build_recover, project_evidence
from terminal_coding_agent.state import ToDoItem, ToDoStatus


def test_evidence_keeps_tool_calls_and_results() -> None:
    messages = [
        HumanMessage(content="fix the import"),
        AIMessage(
            content="",
            tool_calls=[{"name": "run_shell", "args": {"command": "pytest -q"}, "id": "1"}],
        ),
        ToolMessage(
            content="exit_code: 1\nImportError",
            tool_call_id="1",
            name="run_shell",
            status="error",
        ),
    ]

    out = project_evidence(messages)

    assert [entry["action"] for entry in out] == ["run_shell", "run_shell"]
    assert out[0]["detail"] == '{"command": "pytest -q"}'
    assert out[1]["detail"] == "error: exit_code: 1\nImportError"


def test_evidence_clips_detail_to_max_chars() -> None:
    messages = [ToolMessage(content="x" * 500, tool_call_id="1", name="read_file")]

    out = project_evidence(messages, max_chars=20)

    assert out == [{"action": "read_file", "detail": "success: " + "x" * 11}]


def test_evidence_keeps_only_last_entries() -> None:
    messages = [
        ToolMessage(content=f"r{index}", tool_call_id=str(index), name="read_file")
        for index in range(5)
    ]

    out = project_evidence(messages, max_entries=2)

    assert [entry["detail"] for entry in out] == ["success: r3", "success: r4"]


def test_evidence_skips_plain_messages() -> None:
    messages = [HumanMessage(content="hi"), AIMessage(content="no tools here")]

    assert project_evidence(messages) == []


def _make_plan_returning(*steps: str):
    """Stub for make_plan: returns fixed pending steps and bumps turns like the real one."""

    def make_plan(state, config):
        return {
            "messages": [AIMessage(content="replanned")],
            "todo_list": [ToDoItem(status=ToDoStatus.PENDING, description=step) for step in steps],
            "turns": state.get("turns", 0) + 1,
        }

    return make_plan


def _state(todo_list, **extra):
    state = {
        "messages": [HumanMessage(content="fix the import")],
        "todo_list": todo_list,
        "turns": 0,
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cost_rmb": 0.0,
        "stop_reason": None,
        "current_task_index": 0,
        "replan_count": 0,
        "blocked_reason": "pytest fails on import",
        "blocked_evidence": [{"action": "run_shell", "detail": '{"command": "pytest -q"}'}],
    }
    state.update(extra)
    return state


def test_recover_keeps_done_prefix_and_replaces_remainder(tmp_path: Path) -> None:
    node = build_recover(_make_plan_returning("inspect import", "patch import"), worktree=tmp_path)
    state = _state(
        [
            ToDoItem(status=ToDoStatus.DONE, description="read the file"),
            ToDoItem(status=ToDoStatus.FAILED, description="fix the import"),
        ]
    )

    out = node(state, RunnableConfig())

    assert [item.description for item in out["todo_list"]] == [
        "read the file",
        "fix the import",
        "inspect import",
        "patch import",
    ]
    assert [item.status for item in out["todo_list"]] == [
        ToDoStatus.DONE,
        ToDoStatus.DEPRECATED,
        ToDoStatus.PENDING,
        ToDoStatus.PENDING,
    ]
    assert out["replan_count"] == 1
    assert out["blocked_reason"] is None
    assert out["blocked_evidence"] == []
    assert out["current_task_index"] is None
    assert out["turns"] == 1  # budget updates produced by make_plan flow through
    assert [message.type for message in out["messages"]] == ["human", "ai"]
    assert "pytest fails on import" in out["messages"][0].content
    assert "run_shell" in out["messages"][0].content
    assert out["messages"][1].content == "replanned"


def test_recover_keeps_a_leading_failed_step(tmp_path: Path) -> None:
    node = build_recover(_make_plan_returning("edit the file"), worktree=tmp_path)
    state = _state(
        [
            ToDoItem(status=ToDoStatus.FAILED, description="run the script"),
            ToDoItem(status=ToDoStatus.PENDING, description="edit later"),
        ]
    )

    out = node(state, RunnableConfig())

    assert [(item.status, item.description) for item in out["todo_list"]] == [
        (ToDoStatus.DEPRECATED, "run the script"),
        (ToDoStatus.PENDING, "edit the file"),
    ]


def test_recover_increments_replan_counter(tmp_path: Path) -> None:
    node = build_recover(_make_plan_returning("try again"), worktree=tmp_path)
    state = _state([ToDoItem(status=ToDoStatus.FAILED, description="old")], replan_count=2)

    assert node(state, RunnableConfig())["replan_count"] == 3


def test_recover_aborts_when_replan_budget_exhausted(tmp_path: Path) -> None:
    def explode(state, config):
        raise AssertionError("planner must not be called once the cap is reached")

    node = build_recover(explode, worktree=tmp_path)
    state = _state([ToDoItem(status=ToDoStatus.FAILED, description="old")], replan_count=3)

    out = node(state, RunnableConfig())

    assert out["stop_reason"] == "recover_exhausted"
    trace = json.loads((tmp_path / ".agent" / "trace.json").read_text(encoding="utf-8"))
    assert trace["stop_reason"] == "recover_exhausted"
    assert trace["todo_list"][0]["status"] == "FAILED"


def test_recover_aborts_when_planner_returns_same_remaining_steps(tmp_path: Path) -> None:
    node = build_recover(_make_plan_returning("fix the import"), worktree=tmp_path)
    state = _state([ToDoItem(status=ToDoStatus.FAILED, description="fix the import")])

    out = node(state, RunnableConfig())

    assert out["stop_reason"] == "recover_no_progress"
    assert "todo_list" not in out


def test_recover_propagates_a_planner_stop_reason(tmp_path: Path) -> None:
    def failing_plan(state, config):
        return {"stop_reason": "planner_error:RuntimeError"}

    node = build_recover(failing_plan, worktree=tmp_path)
    state = _state([ToDoItem(status=ToDoStatus.FAILED, description="old")])

    out = node(state, RunnableConfig())

    assert out["stop_reason"] == "planner_error:RuntimeError"


def test_evidence_honours_a_non_positive_max_entries() -> None:
    """`entries[-0:]` is the whole list, so 0 must mean "keep nothing"."""
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "run_shell", "args": {"command": "pytest -q"}, "id": "1"}],
        )
    ]

    assert project_evidence(messages, max_entries=0) == []
