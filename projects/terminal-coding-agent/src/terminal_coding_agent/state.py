"""Agent state schema and domain models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import MessagesState
from pydantic import BaseModel, Field


class ToDoStatus(Enum):
    """Lifecycle status of a single todo."""

    PENDING = auto()
    IN_PROGRESS = auto()
    DONE = auto()
    FAILED = auto()


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


def print_todos(todos: list[ToDoItem]) -> None:
    markers: dict[ToDoStatus, str] = {
        ToDoStatus.PENDING: "[-]",
        ToDoStatus.IN_PROGRESS: "[+]",
        ToDoStatus.DONE: "[✓]",
        ToDoStatus.FAILED: "[✗]",
    }
    print("------ Execute State Update --------")
    for todo in todos:
        print(f"{markers[todo.status]}  {todo.description}")
    print("------------------------------------")


def replace_todos(old: list[ToDoItem], new: list[ToDoItem]) -> list[ToDoItem]:
    if _todo_snapshot(old) != _todo_snapshot(new):
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
    cost_usd: float
    stop_reason: str | None = None


def usage_tokens(message: BaseMessage) -> int:
    """Total tokens reported by the provider; 0 when usage is absent."""
    metadata = getattr(message, "usage_metadata", None) or {}
    return metadata.get("total_tokens", 0)
