"""Recover: the executor reports a blocked task and the agent loop stops."""

from typing import Any, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool


@tool
def report_blocked(reason: str) -> str:
    """Report that the current step cannot be completed.

    Call this instead of guessing or retrying the same failing command.
    Include the concrete failing command and its error.

    Args:
        reason: Why the step is blocked.

    Returns:
        An acknowledgement. The harness stops the turn before this body runs.
    """
    return f"Reported blocked: {reason}"


class BlockedState(AgentState):
    """Agent state extended with the executor's block report."""

    blocked_reason: NotRequired[str]


class BlockedReportMiddleware(AgentMiddleware):
    """Pre-empt the tool node: a report_blocked call ends the turn immediately."""

    state_schema = BlockedState

    def __init__(self) -> None:
        super().__init__()
        self.tools = [report_blocked]

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: BlockedState, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        last_ai = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )
        if last_ai is None or not last_ai.tool_calls:
            return None

        blocked = [call for call in last_ai.tool_calls if call["name"] == "report_blocked"]
        if not blocked:
            return None

        reason = str((blocked[0].get("args") or {}).get("reason") or "")
        skipped = [call for call in last_ai.tool_calls if call["name"] != "report_blocked"]
        return {
            "jump_to": "end",
            "blocked_reason": reason,
            "messages": [
                ToolMessage(
                    content="Blocked reported.",
                    tool_call_id=call["id"],
                    name="report_blocked",
                )
                for call in blocked
            ]
            + [
                ToolMessage(
                    content="Skipped: the turn stopped before this call ran.",
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
                for call in skipped
            ]
            + [AIMessage(content=f"Blocked: {reason}")],
        }

    @hook_config(can_jump_to=["end"])
    async def aafter_model(self, state: BlockedState, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)
