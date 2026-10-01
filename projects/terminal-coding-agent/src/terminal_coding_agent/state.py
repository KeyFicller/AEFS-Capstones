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


_MARKERS: dict[ToDoStatus, str] = {
    ToDoStatus.PENDING: "[-]",
    ToDoStatus.IN_PROGRESS: "[+]",
    ToDoStatus.DONE: "[✓]",
    ToDoStatus.FAILED: "[✗]",
    ToDoStatus.DEPRECATED: "[~]",
}


def format_todos(todos: list[ToDoItem], version: int = 0) -> str:
    """Render the todo list as text. Pure; the caller decides where to print it."""
    title = f" Execute State Update | replan v{version} "
    side = "#" * 4
    header = f"{side}{title}{side}"
    body = [f"{_MARKERS[item.status]}  {item.description}" for item in todos]
    return "\n".join([header, *body, "#" * len(header)])


def replace_todos(old: list[ToDoItem], new: list[ToDoItem]) -> list[ToDoItem]:
    """Reducer: newest snapshot wins. Pure, so checkpoint replay is deterministic."""
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
