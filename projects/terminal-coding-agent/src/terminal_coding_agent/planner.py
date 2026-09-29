"""Plan stage: subgraph that splits a task into todo steps."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from terminal_coding_agent.models import AgentModels
from terminal_coding_agent.state import (
    CodingAgentState,
    Plan,
    ToDoItem,
    ToDoStatus,
    usage_tokens,
)


def build_planner(models: AgentModels) -> CompiledStateGraph:
    """Build the planning subgraph: one structured-output call that yields the todo list."""
    planner_model = models.planner.with_structured_output(Plan, include_raw=True)

    def make_plan(state: CodingAgentState) -> dict:
        response = planner_model.invoke(state["messages"])
        parsed: Plan = response["parsed"]

        def format_steps(plan: Plan) -> str:
            return f"Task: {plan.task}\n" + "\n".join(f"- {step}" for step in plan.steps)

        return {
            "messages": [AIMessage(content=format_steps(parsed))],
            "todo_list": [
                ToDoItem(status=ToDoStatus.PENDING, description=step) for step in parsed.steps
            ],
            "turns": state.get("turns", 0) + 1,
            "tokens": state.get("tokens", 0) + usage_tokens(response["raw"]),
        }

    planner = StateGraph(CodingAgentState)
    planner.add_node("make_plan", make_plan)
    planner.add_edge(START, "make_plan")
    planner.add_edge("make_plan", END)

    return planner.compile()
