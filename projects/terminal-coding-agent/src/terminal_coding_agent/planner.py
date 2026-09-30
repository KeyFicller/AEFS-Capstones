"""Plan stage: one structured-output call that yields the todo list."""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage

from terminal_coding_agent.models import AgentModels
from terminal_coding_agent.state import (
    CodingAgentState,
    Plan,
    ToDoItem,
    ToDoStatus,
    usage_tokens,
)
from terminal_coding_agent.telemetry.chat import chat_span, record_chat_usage, resolve_model_name


def build_planner(models: AgentModels) -> Callable[[CodingAgentState], dict[str, Any]]:
    """Return a plain node function (not a nested graph) to avoid double replace_todos."""
    planner_model = models.planner.with_structured_output(Plan, include_raw=True)
    model_name = resolve_model_name(models.planner)

    def make_plan(state: CodingAgentState) -> dict[str, Any]:
        with chat_span(model_name) as span:
            response = planner_model.invoke(state["messages"])
            record_chat_usage(span, response["raw"], model=model_name)
        parsed: Plan = response["parsed"]

        def format_steps(plan: Plan) -> str:
            return f"Task: {plan.task}\n" + "\n".join(f"- {step}" for step in plan.steps)

        return {
            "messages": [AIMessage(content=format_steps(parsed))],
            "todo_list": [
                ToDoItem(status=ToDoStatus.PENDING, description=step)
                for step in parsed.steps
            ],
            "turns": state.get("turns", 0) + 1,
            "tokens": state.get("tokens", 0) + usage_tokens(response["raw"]),
        }

    return make_plan
