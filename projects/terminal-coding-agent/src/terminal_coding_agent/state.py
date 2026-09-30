"""Agent state schema and domain models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import MessagesState
from pydantic import BaseModel, Field


class ToDoStatus(Enum):
    """Lifecycle status of a single todo."""

    PENDING = auto()
    IN_PROGRESS = auto()
    DONE = auto()
    FAILED = auto()
    DEPRECATED = auto()


@dataclass
class ToDoItem:
    """One plan step and its execution status."""

    status: ToDoStatus = ToDoStatus.PENDING
    description: str = ""


class Plan(BaseModel):
    task: str = Field(description="The task to complete")
    steps: list[str] = Field(description="The steps to complete the task")


def _todo_snapshot(todos: list[ToDoItem]) -> list[tuple[str, str]]:
    """Value snapshot so in-place mutations cannot hide updates from equality checks."""
    return [(todo.description, todo.status.name) for todo in todos]


class _VersionedTodos(list):
    """Todo list plus the plan version, so the reducer can print it.

    LangGraph runs the node inside copy_context(), so a ContextVar set in the
    node is invisible when replace_todos later prints.
    """

    replan_version: int


def tag_replan_version(todos: list[ToDoItem], version: int) -> list[ToDoItem]:
    tagged = _VersionedTodos(todos)
    tagged.replan_version = version
    return tagged


def print_todos(todos: list[ToDoItem]) -> None:
    markers: dict[ToDoStatus, str] = {
        ToDoStatus.PENDING: "[-]",
        ToDoStatus.IN_PROGRESS: "[+]",
        ToDoStatus.DONE: "[✓]",
        ToDoStatus.FAILED: "[✗]",
        ToDoStatus.DEPRECATED: "[~]",
    }
    version = getattr(todos, "replan_version", 0)
    title = f" Execute State Update | replan v{version} "
    side = "#" * 4
    header = f"{side}{title}{side}"
    print(header)
    for todo in todos:
        print(f"{markers[todo.status]}  {todo.description}")
    print("#" * len(header))


# Conditional edges read state with fresh=True, which runs this reducer on a
# channel copy. apply_writes then runs it again. Remember the last print so
# that second call does not repeat the same list.
_last_printed: tuple[tuple[str, str], ...] | None = None


def replace_todos(old: list[ToDoItem], new: list[ToDoItem]) -> list[ToDoItem]:
    global _last_printed
    old_snap = tuple(_todo_snapshot(old))
    new_snap = tuple(_todo_snapshot(new))
    if new_snap != old_snap and new_snap != _last_printed:
        _last_printed = new_snap
        print_todos(new)
    return new


class CodingAgentState(MessagesState):
    """State shared by the top-level graph and its subgraphs."""

    todo_list: Annotated[list[ToDoItem], replace_todos]
    turns: int
    tokens: int
    current_task_index: int | None = None

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_rmb: float
    stop_reason: str | None = None

    replan_count: int = 0
    blocked_reason: str | None = None
    blocked_evidence: list[dict[str, Any]] = []


def usage_tokens(message: BaseMessage) -> int:
    """Total tokens reported by the provider; 0 when usage is absent."""
    metadata = getattr(message, "usage_metadata", None) or {}
    return metadata.get("total_tokens", 0)
