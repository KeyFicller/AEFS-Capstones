"""Execute stage: start_task -> run_agent -> end_task (parent-graph nodes)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from terminal_coding_agent.budget import (
    budget_updates,
    check,
    ledger_from_state,
    summarize_todos,
    write_trace,
)
from terminal_coding_agent.config import PRECOMPACT_TOKENS
from terminal_coding_agent.middleware.budget import BudgetMiddleware
from terminal_coding_agent.middleware.observability import ObservabilityMiddleware
from terminal_coding_agent.middleware.safety import SafetyMiddleware
from terminal_coding_agent.models import SYSTEM_PROMPTS, AgentModels
from terminal_coding_agent.state import CodingAgentState, ToDoStatus
from terminal_coding_agent.telemetry.chat import resolve_model_name


def _trace_path(worktree: Path) -> Path:
    return worktree / ".agent" / "trace.json"


def _final_agent_message(agent_messages: list) -> AIMessage | None:
    """Keep only the last AI reply (no tool_calls) for the parent message trail."""
    for message in reversed(agent_messages):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            return message
    return None


def build_execute_nodes(
    models: AgentModels, tools: list, *, worktree: Path
) -> dict[str, Callable[[CodingAgentState], dict[str, Any]]]:
    """Three parent-graph nodes so IN_PROGRESS is committed (and printed) before the agent runs."""

    trace = _trace_path(worktree)

    def start_task(state: CodingAgentState) -> dict[str, Any]:
        ledger = ledger_from_state(state)
        todos = summarize_todos(state.get("todo_list") or [])

        if reason := check(ledger):
            ledger.stop_reason = reason
            write_trace(trace, ledger, todo_list=todos, stop_reason=reason)
            return {**budget_updates(ledger), "stop_reason": reason}

        task_index = next(
            (
                i
                for i, item in enumerate(state["todo_list"])
                if item.status == ToDoStatus.PENDING
            ),
            None,
        )
        if task_index is None:
            raise AssertionError("No pending task found")

        todo_list = list(state["todo_list"])
        todo_list[task_index] = replace(
            todo_list[task_index], status=ToDoStatus.IN_PROGRESS
        )
        return {
            "todo_list": todo_list,
            "current_task_index": task_index,
        }

    def run_agent(state: CodingAgentState) -> dict[str, Any]:
        if state.get("stop_reason"):
            return {}

        task_index = state.get("current_task_index")
        if task_index is None:
            raise AssertionError("current_task_index missing; start_task must run first")

        ledger = ledger_from_state(state)
        todo_list = list(state["todo_list"])
        todos = summarize_todos(todo_list)

        agent = create_agent(
            model=models.executor,
            tools=tools,
            system_prompt=SYSTEM_PROMPTS["executor"],
            middleware=[
                BudgetMiddleware(ledger, trace_path=trace, todo_list=todos),
                SummarizationMiddleware(
                    models.executor, trigger=("tokens", PRECOMPACT_TOKENS)
                ),
                ObservabilityMiddleware(resolve_model_name(models.executor)),
                SafetyMiddleware(),
            ],
        )
        result = agent.invoke(
            {"messages": [HumanMessage(content=todo_list[task_index].description)]}
        )
        agent_messages = result.get("messages") if isinstance(result, dict) else []
        final = _final_agent_message(list(agent_messages))
        updates: dict[str, Any] = {**budget_updates(ledger)}
        if final is not None:
            updates["messages"] = [final]
        return updates

    def end_task(state: CodingAgentState) -> dict[str, Any]:
        ledger = ledger_from_state(state)
        task_index = state.get("current_task_index")
        todo_list = list(state.get("todo_list") or [])

        if task_index is not None and todo_list:
            final_status = ToDoStatus.FAILED if ledger.stop_reason else ToDoStatus.DONE
            todo_list[task_index] = replace(todo_list[task_index], status=final_status)
            current = todo_list[task_index]
            outcome = (
                "successfully completed"
                if current.status == ToDoStatus.DONE
                else "failed"
            )
            write_trace(
                trace,
                ledger,
                todo_list=summarize_todos(todo_list),
                stop_reason=ledger.stop_reason or "completed",
            )
            return {
                **budget_updates(ledger),
                "todo_list": todo_list,
                "messages": [
                    AIMessage(content=f"Task {current.description} {outcome}.")
                ],
                "current_task_index": None,
            }

        return {
            **budget_updates(ledger),
            "current_task_index": None,
        }

    return {
        "start_task": start_task,
        "run_agent": run_agent,
        "end_task": end_task,
    }
