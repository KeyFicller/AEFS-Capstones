"""Execute stage: one agent that plans with `TodoListMiddleware` and carries the task out."""

import logging
from collections.abc import Callable
from typing import Any

from budget import Budget
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
)
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp
from telemetry import resolve_model_name

from terminal_coding_agent.config import MAX_TURNS, PRECOMPACT_TOKENS
from terminal_coding_agent.ledger import budget_updates, ledger_from_state
from terminal_coding_agent.middleware import (
    AskUserMiddleware,
    BlockedReportMiddleware,
    BudgetMiddleware,
    ObservabilityMiddleware,
    SafetyMiddleware,
    SequenceMiddleware,
    TodoMirrorMiddleware,
)
from terminal_coding_agent.models import SYSTEM_PROMPTS, AgentModels
from terminal_coding_agent.state import CodingAgentState

logger = logging.getLogger(__name__)


def _final_agent_message(agent_messages: list) -> AIMessage | None:
    """Keep only the last AI reply (no tool_calls) for the parent message trail.

    A truncated call sits in `invalid_tool_calls` with `tool_calls` empty. Leaving
    it in the parent transcript makes the next model request fail, so the text is
    kept and the broken call is dropped.
    """
    for message in reversed(agent_messages):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            if getattr(message, "invalid_tool_calls", None):
                return message.model_copy(update={"invalid_tool_calls": []})
            return message
    return None


def _executor_prompt(
    *,
    enable_debug: bool,
    shell_anything: bool,
    allow_dangerous: bool = False,
    executor_only: bool = False,
    planning: bool = True,
) -> str:
    """Executor system prompt. Shell-only mode replaces the debug addendum."""
    prompt = SYSTEM_PROMPTS["executor"]
    if shell_anything:
        prompt += "\n\n" + SYSTEM_PROMPTS["shell_anything"]
    elif enable_debug:
        prompt += "\n\n" + SYSTEM_PROMPTS["debug"]
    if allow_dangerous:
        prompt += "\n\n" + SYSTEM_PROMPTS["unrestricted"]
    if executor_only:
        # No `TodoListMiddleware` in this arm, so `write_todos` does not exist.
        prompt += "\n\n" + SYSTEM_PROMPTS["executor_only"]
    elif planning:
        prompt += "\n\n" + SYSTEM_PROMPTS["planning"]
    return prompt


def build_execute_nodes(
    models: AgentModels,
    tools: list,
    *,
    sequence_events: list | None = None,
    enable_debug: bool = False,
    enable_hitl: bool = False,
    shell_anything: bool = False,
    allow_dangerous: bool = False,
    executor_only: bool = False,
    todo_render: Callable[[str], None] | None = None,
) -> dict[str, Callable[[CodingAgentState, RunnableConfig], dict[str, Any]]]:
    """Return the `run_agent` and `chat` nodes plus a compiled agent for diagramming."""

    def _stack(ledger: Any, *, todos: bool, cap: bool, summarize: bool, task: str = "") -> list:
        """Middleware order is meaningful: the first registered sees the model request first."""
        middleware: list = []
        if cap:
            middleware.append(ModelCallLimitMiddleware(run_limit=MAX_TURNS, exit_behavior="error"))
        if todos:
            middleware.append(TodoListMiddleware())
        if summarize:
            middleware.append(
                SummarizationMiddleware(models.executor, trigger=("tokens", PRECOMPACT_TOKENS))
            )
        middleware.append(BudgetMiddleware(ledger))
        middleware.append(ObservabilityMiddleware(resolve_model_name(models.executor)))
        if not allow_dangerous:
            middleware.append(SafetyMiddleware())
        middleware.append(BlockedReportMiddleware())
        if enable_hitl:
            middleware.append(AskUserMiddleware())
        if sequence_events is not None:
            middleware.append(SequenceMiddleware(sequence_events, description=task))
        if todos and todo_render is not None:
            middleware.append(TodoMirrorMiddleware(todo_render))
        return middleware

    def _agent(
        *,
        ledger: Any,
        todos: bool,
        cap: bool,
        summarize: bool,
        task: str,
        system_prompt: str,
    ) -> Any:
        return create_agent(
            model=models.executor,
            tools=tools,
            system_prompt=system_prompt,
            middleware=_stack(ledger, todos=todos, cap=cap, summarize=summarize, task=task),
        )

    def diagram_agent() -> Any:
        """The planning agent, compiled but never invoked: `graph.png` expands it."""
        return _agent(
            ledger=Budget(),
            todos=not executor_only,
            cap=True,
            summarize=True,
            task="",
            system_prompt=_executor_prompt(
                enable_debug=enable_debug,
                shell_anything=shell_anything,
                allow_dangerous=allow_dangerous,
                executor_only=executor_only,
            ),
        )

    def run_agent(state: CodingAgentState, config: RunnableConfig) -> dict[str, Any]:
        """One agent run for the whole request. No per-step orchestration above it."""
        if state.get("stop_reason"):
            return {}

        ledger = ledger_from_state(state)
        task = next(
            (
                message.content
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage) and isinstance(message.content, str)
            ),
            "",
        )
        agent = _agent(
            ledger=ledger,
            todos=not executor_only,
            cap=True,
            summarize=True,
            task=task,
            system_prompt=_executor_prompt(
                enable_debug=enable_debug,
                shell_anything=shell_anything,
                allow_dangerous=allow_dangerous,
                executor_only=executor_only,
            ),
        )

        try:
            result = agent.invoke({"messages": list(state["messages"])}, config=config)
        except GraphBubbleUp:
            # `interrupt()` inside the sub-agent must reach the parent graph instead of being
            # swallowed below into an executor_error. Must precede the broad handler.
            raise
        except ModelCallLimitExceededError:
            return {**budget_updates(ledger), "stop_reason": "max_turns"}
        except Exception as exc:  # noqa: BLE001 - an error is an observation, not a crash
            logger.exception("executor failed")
            return {
                **budget_updates(ledger),
                "stop_reason": f"executor_error:{type(exc).__name__}",
            }

        agent_messages = list(result.get("messages") or [])
        updates: dict[str, Any] = {**budget_updates(ledger)}
        if todo_list := result.get("todo_list"):
            updates["todo_list"] = todo_list

        blocked_reason = result.get("blocked_reason")
        if blocked_reason:
            # No plan stage above to translate a block report, so it ends the turn here.
            updates["stop_reason"] = "blocked"
            updates["blocked_reason"] = blocked_reason
            return updates

        final = _final_agent_message(agent_messages)
        if final is not None:
            updates["messages"] = [final]
        return updates

    def chat(state: CodingAgentState, config: RunnableConfig) -> dict[str, Any]:
        if state.get("stop_reason"):
            return {}

        ledger = ledger_from_state(state)
        agent = _agent(
            ledger=ledger,
            todos=False,
            cap=False,
            summarize=False,
            task="",
            system_prompt=(
                _executor_prompt(
                    enable_debug=False, shell_anything=shell_anything, planning=False
                )
                + "\n\n"
                + SYSTEM_PROMPTS["chat"]
            ),
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

    return {"run_agent": run_agent, "chat": chat, "diagram_agent": diagram_agent}
