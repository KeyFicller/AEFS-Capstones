"""Top-level assembly: Plan -> start_task -> run_agent -> end_task -> Summary."""

from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from terminal_coding_agent.config import ENV_PATH, load_local_env
from terminal_coding_agent.demo import DEMO_TASK, seed_demo_worktree
from terminal_coding_agent.executor import build_execute_nodes
from terminal_coding_agent.models import build_models
from terminal_coding_agent.planner import build_planner
from terminal_coding_agent.state import CodingAgentState, ToDoStatus, usage_tokens
from terminal_coding_agent.tools import make_tools


def make_graph(config: RunnableConfig) -> CompiledStateGraph:
    """Assemble the top-level graph with IN_PROGRESS committed before the agent runs."""
    models = build_models(config)

    raw = (config.get("configurable") or {}).get("worktree")
    worktree = Path(raw).resolve() if raw else Path.cwd().resolve()

    # Plain function node (not a nested StateGraph) so replace_todos does not fire twice.
    make_plan = build_planner(models)
    execute = build_execute_nodes(models, make_tools(worktree), worktree=worktree)

    coding_agent = StateGraph(CodingAgentState)

    def after_plan_or_task(state: CodingAgentState) -> str:
        if state.get("stop_reason"):
            return "summary"
        if not state.get("todo_list"):
            return "summary"
        if all(task.status == ToDoStatus.DONE for task in state["todo_list"]) or any(
            task.status == ToDoStatus.FAILED for task in state["todo_list"]
        ):
            return "summary"
        return "start_task"

    def after_start(state: CodingAgentState) -> str:
        if state.get("stop_reason") or state.get("current_task_index") is None:
            return "summary"
        return "run_agent"

    def summary(state: CodingAgentState) -> dict:
        summary_message = HumanMessage(content="Summarize the task.")
        summary_input = state["messages"] + [summary_message]
        summary_response = models.planner.invoke(summary_input)
        return {
            "messages": [summary_message, summary_response],
            "turns": state.get("turns", 0) + 1,
            "tokens": state.get("tokens", 0) + usage_tokens(summary_response),
        }

    coding_agent.add_node("make_plan", make_plan)
    coding_agent.add_node("start_task", execute["start_task"])
    coding_agent.add_node("run_agent", execute["run_agent"])
    coding_agent.add_node("end_task", execute["end_task"])
    coding_agent.add_node("summary", summary)

    coding_agent.add_edge(START, "make_plan")
    coding_agent.add_conditional_edges(
        "make_plan",
        after_plan_or_task,
        {"start_task": "start_task", "summary": "summary"},
    )
    coding_agent.add_conditional_edges(
        "start_task",
        after_start,
        {"run_agent": "run_agent", "summary": "summary"},
    )
    coding_agent.add_edge("run_agent", "end_task")
    coding_agent.add_conditional_edges(
        "end_task",
        after_plan_or_task,
        {"start_task": "start_task", "summary": "summary"},
    )
    coding_agent.add_edge("summary", END)

    return coding_agent.compile()


if __name__ == "__main__":
    load_local_env(ENV_PATH)
    with tempfile.TemporaryDirectory() as temp_dir:
        worktree = Path(temp_dir)
        seed_demo_worktree(worktree)
        agent = make_graph({"configurable": {"worktree": str(worktree)}})
        response = agent.invoke({"messages": [HumanMessage(content=DEMO_TASK)]})
        for message in response["messages"]:
            message.pretty_print()
        print("------ Budget --------")
        print(f"turns:             {response.get('turns', 0)}")
        print(f"tokens:            {response.get('tokens', 0)}")
        print(f"input_tokens:      {response.get('input_tokens', 0)}")
        print(f"output_tokens:     {response.get('output_tokens', 0)}")
        print(f"cache_read_tokens: {response.get('cache_read_tokens', 0)}")
        print(f"cost_usd:          {response.get('cost_usd', 0.0):.6f}")
        print(f"stop_reason:       {response.get('stop_reason')}")
        print("----------------------")
