"""Plan stage: one structured-output call that yields the todo list."""

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from terminal_coding_agent.budget import BudgetSession
from terminal_coding_agent.models import SYSTEM_PROMPTS, AgentModels
from terminal_coding_agent.state import (
    CodingAgentState,
    Plan,
    RoutingPlan,
    ToDoItem,
    ToDoStatus,
)
from terminal_coding_agent.telemetry import (
    chat_span,
    record_chat_usage,
    resolve_model_name,
)

logger = logging.getLogger(__name__)


def build_planner(
    models: AgentModels, *, enable_answer: bool = False
) -> Callable[..., dict[str, Any]]:
    """Return a plain node function (not a nested graph) to avoid double replace_todos."""
    schema = RoutingPlan if enable_answer else Plan
    planner_model = models.planner.with_structured_output(schema, include_raw=True)
    model_name = resolve_model_name(models.planner)

    def make_plan(state: CodingAgentState, config: RunnableConfig) -> dict[str, Any]:
        error: str | None = None
        domain: dict[str, Any] = {}
        with BudgetSession(state) as budget:
            try:
                with chat_span(model_name) as span:
                    messages = list(state["messages"])
                    if enable_answer:
                        messages = [SystemMessage(content=SYSTEM_PROMPTS["planner"]), *messages]
                    response = planner_model.invoke(messages, config=config)
                    record_chat_usage(span, response["raw"], model=model_name)
                budget.observe(response["raw"])
                parsed = response["parsed"]
                if parsed is None:
                    raise ValueError(f"structured output parse failed: {response['parsing_error']}")

                if getattr(parsed, "mode", "work") == "answer":
                    domain = {"mode": "answer"}
                else:
                    domain = {
                        "mode": "work",
                        "messages": [
                            AIMessage(
                                content=f"Task: {parsed.task}\n"
                                + "\n".join(f"- {step}" for step in parsed.steps)
                            )
                        ],
                        "todo_list": [
                            ToDoItem(status=ToDoStatus.PENDING, description=step)
                            for step in parsed.steps
                        ],
                    }
            except Exception as exc:  # noqa: BLE001 - an error is an observation
                logger.exception("planner failed")
                error = f"planner_error:{type(exc).__name__}"
        if error is not None:
            return {**budget.updates(), "stop_reason": error}
        return {**domain, **budget.updates()}

    return make_plan
