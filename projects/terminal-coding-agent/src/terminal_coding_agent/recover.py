"""Recover stage: bounded replanning after a task reports itself blocked."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from itertools import takewhile
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from terminal_coding_agent.budget import (
    budget_updates,
    ledger_from_state,
    summarize_todos,
    trace_path,
    write_trace,
)
from terminal_coding_agent.config import (
    EVIDENCE_MAX_CHARS,
    EVIDENCE_MAX_ENTRIES,
    MAX_REPLANS,
)
from terminal_coding_agent.state import (
    CodingAgentState,
    ToDoItem,
    ToDoStatus,
)


def project_evidence(
    messages: Sequence[BaseMessage],
    *,
    max_entries: int = EVIDENCE_MAX_ENTRIES,
    max_chars: int = EVIDENCE_MAX_CHARS,
) -> list[dict[str, str]]:
    """Compact transcript for replanning: one entry per tool call or tool result.

    Entry: {"action": <tool name>, "detail": <args json | "<status>: <content>">}.
    Human messages and AI messages without tool calls are skipped; only the last
    `max_entries` survive; every `detail` is clipped to `max_chars`.

    `detail` is always a string: a tool result whose `content` is a list of content
    blocks is stringified first, so `max_chars` bounds the rendered text rather than
    the original blocks.
    """

    entries: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                entries.append(
                    {
                        "action": call["name"],
                        "detail": json.dumps(
                            call.get("args") or {}, ensure_ascii=False, sort_keys=True
                        ),
                    }
                )
        elif isinstance(message, ToolMessage):
            entries.append(
                {
                    "action": message.name or "tool",
                    "detail": f"{message.status}: {message.content}",
                }
            )
    tail = entries[-max_entries:]
    return [{**entry, "detail": entry["detail"][:max_chars]} for entry in tail]


def _replan_instruction(state: CodingAgentState) -> str:
    failed = next(
        (
            item.description
            for item in state.get("todo_list") or []
            if item.status == ToDoStatus.FAILED
        ),
        "(unknown)",
    )

    evidence = "\n".join(
        f"- {entry['action']}  {entry['detail']}"
        for entry in state.get("blocked_evidence") or []
    )

    return (
        "The step below failed and could not be completed.\n\n"
        f"Failed step: {failed}\n"
        f"Reason: {state.get('blocked_reason') or '(none)'}\n"
        f"Attempts (tool call, then result):\n{evidence}\n\n"
        "Rewrite the remaining steps so the overall task can still be completed. "
        "You may split, reorder, replace, or drop steps. If the failed step left files "
        "in a bad state, include a step to revert it (use the git or edit_file tools). "
        "Do not repeat work that is already done."
    )


_SETTLED = {ToDoStatus.DONE, ToDoStatus.DEPRECATED}


def _history_prefix(todo_list: Sequence[ToDoItem]) -> list[ToDoItem]:
    """Keep finished steps, and retire the FAILED step as DEPRECATED."""
    kept = list(takewhile(lambda item: item.status in _SETTLED, todo_list))
    rest = todo_list[len(kept) :]
    if rest and rest[0].status == ToDoStatus.FAILED:
        return [*kept, replace(rest[0], status=ToDoStatus.DEPRECATED)]
    return kept


def _remaining_descriptions(todo_list: Sequence[ToDoItem]) -> list[str]:
    """The descriptions of steps that replanning must still complete."""
    return [item.description for item in todo_list if item.status not in _SETTLED]


def _abort(state: CodingAgentState, worktree: Path, reason: str) -> dict[str, Any]:
    ledger = ledger_from_state(state)
    write_trace(
        trace_path(worktree),
        ledger,
        todo_list=summarize_todos(state.get("todo_list") or []),
        stop_reason=reason,
    )
    return {**budget_updates(ledger), "stop_reason": reason}


def build_recover(
    make_plan: Callable[..., dict[str, Any]], *, worktree: Path
) -> Callable[[CodingAgentState, RunnableConfig], dict[str, Any]]:
    """Return the recover node: bounded replan of the remaining todos."""

    def recover(state: CodingAgentState, config: RunnableConfig) -> dict[str, Any]:
        todo_list = list(state.get("todo_list") or [])
        replan_count = state.get("replan_count", 0)

        if replan_count >= MAX_REPLANS:
            return _abort(state, worktree=worktree, reason="recover_exhausted")

        instruction = HumanMessage(content=_replan_instruction(state))
        planned = make_plan({**state, "messages": [*state["messages"], instruction]}, config)
        if planned.get("stop_reason"):
            return planned
        new_steps = list(planned.get("todo_list") or [])
        new_descriptions = [item.description for item in new_steps]
        if new_descriptions == _remaining_descriptions(todo_list):
            return _abort(state, worktree=worktree, reason="recover_no_progress")

        merged = _history_prefix(todo_list) + new_steps

        return {
            **planned,
            "todo_list": merged,
            "messages": [instruction, *planned.get("messages", [])],
            "replan_count": replan_count + 1,
            "blocked_reason": None,
            "blocked_evidence": [],
            "current_task_index": None,
        }

    return recover
