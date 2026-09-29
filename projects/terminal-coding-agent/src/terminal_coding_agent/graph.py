"""Top-level assembly: the CodingAgent graph (Plan -> Execute -> Summary)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from terminal_coding_agent.config import ENV_PATH, load_local_env
from terminal_coding_agent.executor import build_executor
from terminal_coding_agent.models import build_models
from terminal_coding_agent.planner import build_planner
from terminal_coding_agent.state import CodingAgentState, ToDoStatus, usage_tokens


def make_graph(config: RunnableConfig) -> CompiledStateGraph:
    """Assemble the top-level graph: make_plan -> execute_task (looped) -> summary."""
    models = build_models(config)
    planner = build_planner(models)
    executor = build_executor(models)

    coding_agent = StateGraph(CodingAgentState)

    def should_continue(state: CodingAgentState) -> str:
        if all(task.status == ToDoStatus.DONE for task in state["todo_list"]) or any(
            task.status == ToDoStatus.FAILED for task in state["todo_list"]
        ):
            return "summary"

        return "execute_task"

    def summary(state: CodingAgentState) -> dict:
        summary_message = HumanMessage(content="Summarize the task.")
        summary_input = state["messages"] + [summary_message]
        # Use the unbound models.planner: the planner subgraph's model is bound to
        # structured output (Plan) and cannot produce free-form text.
        summary_response = models.planner.invoke(summary_input)
        return {
            "messages": [summary_message, summary_response],
            "turns": state.get("turns", 0) + 1,
            "tokens": state.get("tokens", 0) + usage_tokens(summary_response),
        }

    coding_agent.add_node("make_plan", planner)
    coding_agent.add_node("execute_task", executor)
    coding_agent.add_node("summary", summary)

    coding_agent.add_edge(START, "make_plan")
    coding_agent.add_conditional_edges(
        "make_plan",
        should_continue,
        {
            "execute_task": "execute_task",
            "summary": "summary",
        },
    )

    coding_agent.add_conditional_edges(
        "execute_task",
        should_continue,
        {
            "execute_task": "execute_task",
            "summary": "summary",
        },
    )

    coding_agent.add_edge("summary", END)

    return coding_agent.compile()


if __name__ == "__main__":
    load_local_env(ENV_PATH)
    agent = make_graph({})
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Create a Python script that prints 'Hello LangChain'. "
                        "Then run the script to veirfy the result."
                    )
                )
            ]
        }
    )

    for message in response["messages"]:
        message.pretty_print()
