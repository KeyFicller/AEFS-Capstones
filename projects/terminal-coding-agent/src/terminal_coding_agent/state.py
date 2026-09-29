"""Agent state schema and domain models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

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


class CodingAgentState(MessagesState):
    """State shared by the top-level graph and its subgraphs."""

    todo_list: list[ToDoItem]
    turns: int
    tokens: int
    current_task_index: int | None = None
    current_task_messages: list[BaseMessage] = []


def usage_tokens(message: BaseMessage) -> int:
    """Total tokens reported by the provider; 0 when usage is absent."""
    metadata = getattr(message, "usage_metadata", None) or {}
    return metadata.get("total_tokens", 0)
