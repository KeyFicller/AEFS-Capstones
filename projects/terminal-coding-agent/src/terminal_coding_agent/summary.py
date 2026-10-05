"""Summary stage — the graph's single Stop point. The trace is written, always."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, message_chunk_to_message
from langchain_core.runnables import RunnableConfig
from telemetry import chat_span, record_chat_usage, resolve_model_name

from terminal_coding_agent.artifacts import publish_patch
from terminal_coding_agent.attachments import strip_images
from terminal_coding_agent.ledger import (
    BudgetSession,
    ledger_from_state,
    summarize_todos,
    trace_path,
    write_trace,
)
from terminal_coding_agent.middleware.sequence import write_sequence_png
from terminal_coding_agent.models import AgentModels
from terminal_coding_agent.state import CodingAgentState

logger = logging.getLogger(__name__)


def _text_of(message: Any) -> str:
    """Text of a chat message; multimodal content contributes only its text blocks."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _answer_reply(messages: list) -> AIMessage | None:
    """Answer mode's deliverable: the last message, when it is non-empty assistant text."""
    if not messages:
        return None
    last = messages[-1]
    if isinstance(last, AIMessage) and _text_of(last).strip():
        return last
    return None


def _stream_summary(
    model: Any,
    messages: list,
    config: RunnableConfig,
    render: Callable[[str], None],
) -> Any:
    """Stream the summary, feeding `render` the whole text so far after each chunk.

    stream_usage is explicit: ChatDeepSeek omits usage when streaming, which would zero
    the summary's budget; ChatOllama lacks the switch, so it is passed conditionally.
    """
    kwargs = {"stream_usage": True} if hasattr(model, "stream_usage") else {}
    accumulated = None
    for chunk in model.stream(messages, config=config, **kwargs):
        accumulated = chunk if accumulated is None else accumulated + chunk
        render(_text_of(accumulated))
    if accumulated is None:
        response = model.invoke(messages, config=config)
        render(_text_of(response))
        return response
    return message_chunk_to_message(accumulated)


def build_summary(
    models: AgentModels,
    *,
    worktree: Path,
    sequence_events: list | None = None,
    sequence_path: Path | None = None,
    summary_renderer: Callable[[str], None] | None = None,
) -> Callable[[CodingAgentState, RunnableConfig], dict[str, Any]]:
    """Return the summary node. Its finally block is the Stop hook: it never skips.

    An `answer`-mode run relays its last message instead of summarizing: the content is
    the deliverable. `summary_renderer` (interactive CLI only) streams the summary call
    and paints the relayed answer; without it, both are one-shot.
    """

    def summary(state: CodingAgentState, config: RunnableConfig) -> dict[str, Any]:
        summary_message = HumanMessage(content="Summarize the task.")
        summary_input = strip_images(list(state["messages"])) + [summary_message]
        model_name = resolve_model_name(models.planner)
        ledger = ledger_from_state(state)
        try:
            with BudgetSession(state) as budget:
                reply = (
                    _answer_reply(list(state["messages"]))
                    if state.get("mode") == "answer"
                    else None
                )
                if reply is not None:
                    if summary_renderer is not None:
                        summary_renderer(_text_of(reply))
                    domain = {"messages": [AIMessage(content=reply.content)]}
                else:
                    with chat_span(model_name) as span:
                        if summary_renderer is not None:
                            response = _stream_summary(
                                models.planner, summary_input, config, summary_renderer
                            )
                        else:
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
