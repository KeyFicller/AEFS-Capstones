"""Interactive session semantics: per-turn reset and turn execution."""

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command


def reset_updates() -> dict[str, Any]:
    """Fresh per-turn state for one task.

    `messages` is deliberately absent: the transcript is the session's memory and
    must accumulate (`add_messages`), while every other field is budget or plan
    bookkeeping whose PROJECT.md budget is *per task*. A stale `stop_reason` would
    make `_after_make_plan` jump straight to `summary` and end the turn instantly.
    Built fresh each call so no two turns share a list.
    """
    return {
        "todo_list": [],
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


def run_task_turn(*, graph: Any, config: dict, text: str) -> dict[str, Any]:
    """Reset per-turn state, then run one full plan->act->observe->recover pass."""
    graph.update_state(config, reset_updates())
    return graph.invoke({"messages": [HumanMessage(content=text)]}, config)


def resume_turn(*, graph: Any, config: dict, decision: str) -> dict[str, Any]:
    """Resume a paused turn. Never `update_state`: the pause *is* the checkpoint."""
    return graph.invoke(Command(resume=decision), config)


def pending_interrupts(graph: Any, config: dict) -> list:
    """Interrupts a paused graph is waiting on; empty when it is not paused."""
    snapshot = graph.get_state(config)
    return [item for task in snapshot.tasks for item in task.interrupts]
