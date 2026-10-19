from typing import Any

from terminal_coding_agent.middleware.todos import TodoMirrorMiddleware, mirror_todos
from terminal_coding_agent.state import ToDoStatus


def test_mirror_maps_native_statuses_to_ours() -> None:
    todos = [
        {"content": "read the README", "status": "completed"},
        {"content": "patch the bug", "status": "in_progress"},
        {"content": "run the tests", "status": "pending"},
    ]

    assert [(item.status, item.description) for item in mirror_todos(todos)] == [
        (ToDoStatus.DONE, "read the README"),
        (ToDoStatus.IN_PROGRESS, "patch the bug"),
        (ToDoStatus.PENDING, "run the tests"),
    ]


def test_mirror_keeps_unknown_statuses_pending() -> None:
    assert mirror_todos([{"content": "x", "status": "weird"}])[0].status == ToDoStatus.PENDING


def test_mirror_ignores_junk_entries() -> None:
    assert mirror_todos(None) == []
    assert mirror_todos(["not a dict"]) == []  # type: ignore[list-item]


def _state(**extra: Any) -> dict[str, Any]:
    return {"messages": [], "todos": [], **extra}


def test_middleware_publishes_todos_and_redraws_the_panel() -> None:
    drawn: list[str] = []
    node = TodoMirrorMiddleware(drawn.append)
    state = _state(todos=[{"content": "fix typo", "status": "in_progress"}])

    out = node.after_model(state, None)

    assert out is not None
    assert out["todo_list"][0].description == "fix typo"
    assert drawn == ["[+]  fix typo"]


def test_middleware_stays_silent_before_the_agent_plans() -> None:
    drawn: list[str] = []
    node = TodoMirrorMiddleware(drawn.append)

    assert node.after_model(_state(), None) is None
    assert drawn == []


def test_middleware_redraws_only_when_the_plan_moves() -> None:
    """One todo spans many model turns; a redraw per turn would flood the Harbor log."""
    drawn: list[str] = []
    node = TodoMirrorMiddleware(drawn.append)
    planned = _state(todos=[{"content": "fix typo", "status": "pending"}])
    done = _state(todos=[{"content": "fix typo", "status": "completed"}])

    node.after_model(planned, None)
    node.after_model(planned, None)
    node.after_model(done, None)

    assert drawn == ["[-]  fix typo", "[✓]  fix typo"]
