"""Summary stage — the graph's single Stop point. The trace is written, always."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from terminal_coding_agent.artifacts import publish_patch
from terminal_coding_agent.budget import (
    BudgetSession,
    ledger_from_state,
    summarize_todos,
    trace_path,
    write_trace,
)
from terminal_coding_agent.middleware.sequence import write_sequence_png
from terminal_coding_agent.models import AgentModels
from terminal_coding_agent.state import CodingAgentState
from terminal_coding_agent.telemetry import chat_span, record_chat_usage, resolve_model_name


def build_summary(
    models: AgentModels,
    *,
    worktree: Path,
    sequence_events: list | None = None,
    sequence_path: Path | None = None,
) -> Callable[[CodingAgentState, RunnableConfig], dict[str, Any]]:
    """Return the summary node. Its finally block is the Stop hook: it never skips."""

    def summary(state: CodingAgentState, config: RunnableConfig) -> dict[str, Any]:
        summary_message = HumanMessage(content="Summarize the task.")
        summary_input = state["messages"] + [summary_message]
        model_name = resolve_model_name(models.planner)
        ledger = ledger_from_state(state)
        try:
            with BudgetSession(state) as budget:
                with chat_span(model_name) as span:
                    response = models.planner.invoke(summary_input, config=config)
                    record_chat_usage(span, response, model=model_name)
                budget.observe(response)
                domain = {"messages": [summary_message, response]}
            ledger = budget.ledger
            return {**domain, **budget.updates()}
        finally:
            # The patch goes first: the Stop hook must publish it even if a
            # later diagnostic write fails.
            publish_patch(worktree)
            write_trace(
                trace_path(worktree),
                ledger,
                todo_list=summarize_todos(state.get("todo_list") or []),
                stop_reason=state.get("stop_reason") or "completed",
            )
            write_sequence_png(sequence_events, sequence_path)

    return summary
