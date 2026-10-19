"""Mirror `TodoListMiddleware`'s todos into the parent-facing `todo_list`.

`TodoListMiddleware` is the planner: its `write_todos` tool keeps the plan inside the
agent. The rest of this project (trace, CLI panel) reads `todo_list`, so this layer maps
one to the other and redraws the panel as the agent revises its plan.
"""

from collections.abc import Callable
from typing import Any, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState

from terminal_coding_agent.state import ToDoItem, ToDoStatus, format_todos

_STATUS = {
    "pending": ToDoStatus.PENDING,
    "in_progress": ToDoStatus.IN_PROGRESS,
    "completed": ToDoStatus.DONE,
}


class TodoMirrorState(AgentState):
    """Agent state extended with the parent-facing todo list."""

    todo_list: NotRequired[list[ToDoItem]]


def mirror_todos(todos: list[dict[str, Any]] | None) -> list[ToDoItem]:
    """Map native todos to `ToDoItem`s. Unknown statuses stay PENDING rather than raise."""
    return [
        ToDoItem(
            status=_STATUS.get(str(todo.get("status")), ToDoStatus.PENDING),
            description=str(todo.get("content") or ""),
        )
        for todo in todos or []
        if isinstance(todo, dict)
    ]


class TodoMirrorMiddleware(AgentMiddleware):
    """Publish the agent's own todo list to `todo_list` and to the console panel."""

    state_schema = TodoMirrorState

    def __init__(self, render: Callable[[str], None]) -> None:
        super().__init__()
        self._render = render
        self._last: list[tuple[ToDoStatus, str]] = []

    def after_model(self, state: TodoMirrorState, runtime: Any) -> dict[str, Any] | None:
        todos = mirror_todos(state.get("todos"))
        if not todos:
            return None
        fingerprint = [(item.status, item.description) for item in todos]
        # The model turns many times per todo; redraw only when the plan actually moved.
        if fingerprint != self._last:
            self._last = fingerprint
            self._render(format_todos(todos))
        return {"todo_list": todos}

    async def aafter_model(self, state: TodoMirrorState, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)
