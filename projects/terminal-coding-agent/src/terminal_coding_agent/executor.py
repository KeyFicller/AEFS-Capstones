"""Execute stage: start_task -> run_agent -> end_task (parent-graph nodes)."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp
from telemetry import resolve_model_name

from terminal_coding_agent.attachments import attached_images
from terminal_coding_agent.config import PRECOMPACT_TOKENS
from terminal_coding_agent.ledger import (
    budget_updates,
    ledger_from_state,
    summarize_todos,
    trace_path,
    write_trace,
)
from terminal_coding_agent.middleware import (
    AskUserMiddleware,
    BlockedReportMiddleware,
    BudgetMiddleware,
    ObservabilityMiddleware,
    SafetyMiddleware,
    SequenceMiddleware,
    ToolLogMiddleware,
)
from terminal_coding_agent.models import SYSTEM_PROMPTS, AgentModels
from terminal_coding_agent.recover import project_evidence
from terminal_coding_agent.state import CodingAgentState, ToDoStatus, format_todos

logger = logging.getLogger(__name__)


def _final_agent_message(agent_messages: list) -> AIMessage | None:
    """Keep only the last AI reply (no tool_calls) for the parent message trail."""
    for message in reversed(agent_messages):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            return message
    return None


def build_execute_nodes(
    models: AgentModels,
    tools: list,
    *,
    worktree: Path,
    sequence_events: list | None = None,
    tool_renderer: Callable[[str, Mapping[str, Any], str], None] | None = None,
    enable_debug: bool = False,
    enable_hitl: bool = False,
) -> dict[str, Callable[[CodingAgentState], dict[str, Any]]]:
    """Three parent-graph nodes so IN_PROGRESS is committed (and printed) before the agent runs."""

    trace = trace_path(worktree)

    def start_task(state: CodingAgentState) -> dict[str, Any]:
        ledger = ledger_from_state(state)
        todos = summarize_todos(state.get("todo_list") or [])

        # A new task starts unblocked: the block report belongs to the task that just ended.
        clear_block = {"blocked_reason": None, "blocked_evidence": []}

        if reason := ledger.check():
            ledger.stop_reason = reason
            write_trace(trace, ledger, todo_list=todos, stop_reason=reason)
            return {**budget_updates(ledger), **clear_block, "stop_reason": reason}

        todo_items = list(state.get("todo_list") or [])
        task_index = next(
            (i for i, item in enumerate(todo_items) if item.status == ToDoStatus.PENDING),
            None,
        )
        if task_index is None:
            raise AssertionError("No pending task found")

        todo_list = list(todo_items)
        todo_list[task_index] = replace(todo_list[task_index], status=ToDoStatus.IN_PROGRESS)
        return {
            "todo_list": todo_list,
            "current_task_index": task_index,
            **clear_block,
        }

    def run_agent(state: CodingAgentState, config: RunnableConfig) -> dict[str, Any]:
        if state.get("stop_reason"):
            return {}

        task_index = state.get("current_task_index")
        if task_index is None:
            raise AssertionError("current_task_index missing; start_task must run first")

        ledger = ledger_from_state(state)
        todo_list = list(state["todo_list"])

        middleware = [
            BudgetMiddleware(ledger),
            SummarizationMiddleware(models.executor, trigger=("tokens", PRECOMPACT_TOKENS)),
            ObservabilityMiddleware(resolve_model_name(models.executor)),
            SafetyMiddleware(),
            BlockedReportMiddleware(),
        ]
        if enable_hitl:
            middleware.append(AskUserMiddleware())
        if sequence_events is not None:
            # Recording is on only when the caller asked for a diagram (sequence_path).
            middleware.append(
                SequenceMiddleware(sequence_events, description=todo_list[task_index].description)
            )
        if tool_renderer is not None:
            # Last = innermost, so a guard that short-circuits the call is not logged as output.
            middleware.append(ToolLogMiddleware(tool_renderer))

        base_prompt = SYSTEM_PROMPTS["executor"]
        if enable_debug:
            base_prompt += "\n\n" + SYSTEM_PROMPTS["debug"]
        system_prompt = (
            base_prompt
            + "\n\nCurrent plan:\n"
            + format_todos(todo_list)
            + f"\n\nYour step: {todo_list[task_index].description}"
        )
        agent = create_agent(
            model=models.executor,
            tools=tools,
            system_prompt=system_prompt,
            middleware=middleware,
        )
        images = attached_images(list(state["messages"]))
        step = todo_list[task_index].description
        step_content: Any = step if not images else [{"type": "text", "text": step}, *images]
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=step_content)]},
                config=config,
            )
        except GraphBubbleUp:
            # `interrupt()` inside the sub-agent must reach the parent graph instead of being
            # swallowed below into an executor_error. Must precede the broad handler.
            raise
        except Exception as exc:  # noqa: BLE001 - an error is an observation, not a crash
            logger.exception("executor failed")
            return {
                **budget_updates(ledger),
                "stop_reason": f"executor_error:{type(exc).__name__}",
            }
        agent_messages = list(result.get("messages") or [])
        updates: dict[str, Any] = {**budget_updates(ledger)}

        blocked_reason = result.get("blocked_reason")
        if blocked_reason:
            updates["blocked_reason"] = blocked_reason
            updates["blocked_evidence"] = project_evidence(agent_messages)
            return updates

        final = _final_agent_message(list(agent_messages))
        if final is not None:
            updates["messages"] = [final]
        return updates

    def chat(state: CodingAgentState, config: RunnableConfig) -> dict[str, Any]:
        if state.get("stop_reason"):
            return {}

        ledger = ledger_from_state(state)
        middleware = [
            BudgetMiddleware(ledger),
            ObservabilityMiddleware(resolve_model_name(models.executor)),
            SafetyMiddleware(),
        ]
        if tool_renderer is not None:
            # Last = innermost, so a guard that short-circuits the call is not logged as output.
            middleware.append(ToolLogMiddleware(tool_renderer))

        agent = create_agent(
            model=models.executor,
            tools=tools,
            system_prompt=SYSTEM_PROMPTS["executor"] + "\n\n" + SYSTEM_PROMPTS["chat"],
            middleware=middleware,
        )
        try:
            result = agent.invoke({"messages": list(state["messages"])}, config=config)
        except GraphBubbleUp:
            raise
        except Exception as exc:  # noqa: BLE001 - an error is an observation, not a crash
            logger.exception("chat failed")
            return {
                **budget_updates(ledger),
                "stop_reason": f"executor_error:{type(exc).__name__}",
            }

        updates: dict[str, Any] = {**budget_updates(ledger)}
        final = _final_agent_message(list(result.get("messages") or []))
        if final is not None:
            updates["messages"] = [final]
        return updates

    def end_task(state: CodingAgentState) -> dict[str, Any]:
        ledger = ledger_from_state(state)
        task_index = state.get("current_task_index")
        todo_list = list(state.get("todo_list") or [])

        if task_index is not None and todo_list:
            blocked = bool(state.get("blocked_reason"))
            final_status = ToDoStatus.FAILED if (ledger.stop_reason or blocked) else ToDoStatus.DONE
            todo_list[task_index] = replace(todo_list[task_index], status=final_status)
            current = todo_list[task_index]
            outcome = "successfully completed" if current.status == ToDoStatus.DONE else "failed"
            write_trace(
                trace,
                ledger,
                todo_list=summarize_todos(todo_list),
                stop_reason=ledger.stop_reason or ("blocked" if blocked else "completed"),
            )
            return {
                **budget_updates(ledger),
                "todo_list": todo_list,
                "messages": [AIMessage(content=f"Task {current.description} {outcome}.")],
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
        "chat": chat,
    }
