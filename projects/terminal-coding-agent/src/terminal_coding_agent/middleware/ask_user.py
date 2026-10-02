"""ask_user: the model asks the human, and the harness pauses before any tool runs.

The interrupt lives in `after_model`, not in the tool body. LangGraph replays a whole node on
resume, so an `interrupt()` inside a tool makes the ToolNode re-run its whole batch, executing
every sibling tool with side effects (`edit_file`, `run_shell`, `git`) twice. Stopping before
the ToolNode means nothing has run yet, so a resume replays nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.types import interrupt

MIN_OPTIONS = 1
MAX_OPTIONS = 4


@tool
def ask_user(question: str, options: list[str]) -> str:
    """Ask the user to choose, when the task is ambiguous or you need their preference.

    Give one to four concrete options. The user may also answer in their own words.

    Args:
        question: The question to put to the user.
        options: Between one and four concrete choices.

    Returns:
        The user's answer. `AskUserMiddleware` answers this call before this body runs.
    """
    return "Error: AskUserMiddleware did not intercept ask_user"


def _answer_text(value: Mapping[str, Any]) -> str:
    """Turn the resume value into the ToolMessage the model reads."""
    if value.get("cancelled"):
        return "The user did not answer; use your best judgement and continue."
    return f"User answered: {value.get('answer')}"


def _error(text: str, call: Mapping[str, Any]) -> ToolMessage:
    return ToolMessage(
        content=text,
        tool_call_id=call["id"],
        name="ask_user",
        status="error",
    )


class AskUserMiddleware(AgentMiddleware):
    """Stop before the tool node: a question is answered here, so no tool ever replays."""

    state_schema = AgentState

    def __init__(self) -> None:
        super().__init__()
        self.tools = [ask_user]

    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        last_ai = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )
        if last_ai is None or not last_ai.tool_calls:
            return None

        asked = [call for call in last_ai.tool_calls if call["name"] == "ask_user"]
        if not asked:
            return None

        # Nothing above `interrupt()` may have a side effect: on resume this re-runs from the top.
        question, extra = asked[0], asked[1:]
        args = question.get("args") or {}
        options = [str(option) for option in args.get("options") or []]

        answers: list[ToolMessage] = []
        if MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
            value = interrupt(
                {
                    "type": "question",
                    "question": str(args.get("question", "")),
                    "options": options,
                }
            )
            answers.append(
                ToolMessage(
                    content=_answer_text(value),
                    tool_call_id=question["id"],
                    name="ask_user",
                )
            )
        else:
            answers.append(
                _error(
                    f"Error: ask_user needs {MIN_OPTIONS} to {MAX_OPTIONS} options; "
                    f"got {len(options)}.",
                    question,
                )
            )
        answers.extend(_error("Error: ask one question per step.", call) for call in extra)

        # The ask_user calls stay in `tool_calls` and are answered here. The agent's router then
        # sees a call whose tool message already exists — no pending call, so the tool node never
        # runs it and the loop goes back to the model instead of ending on "no tool calls".
        return {"messages": answers}

    async def aafter_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)
