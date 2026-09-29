"""Execute stage: act / observe / recover subgraph that runs todos one by one."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from terminal_coding_agent.models import SYSTEM_PROMPTS, AgentModels
from terminal_coding_agent.state import CodingAgentState, ToDoStatus, usage_tokens
from terminal_coding_agent.tools import create_file, run_script, verify_script

TOOLS_BY_NAME = {
    "create_file": create_file,
    "run_script": run_script,
    "verify_script": verify_script,
}


def build_executor(models: AgentModels) -> CompiledStateGraph:
    """Build the execution subgraph: start_task -> (ask_llm <-> use_tool)* -> end_task."""
    executor_model = models.executor.bind_tools(list(TOOLS_BY_NAME.values()))

    def start_task(state: CodingAgentState) -> dict:
        task_index = next(
            (i for i, item in enumerate(state["todo_list"]) if item.status == ToDoStatus.PENDING),
            None,
        )
        if task_index is None:
            raise AssertionError("No pending task found")

        todo_list = list(state["todo_list"])
        todo_list[task_index].status = ToDoStatus.IN_PROGRESS
        messages = [
            SystemMessage(content=SYSTEM_PROMPTS["executor"]),
            HumanMessage(content=todo_list[task_index].description),
        ]
        return {
            "todo_list": todo_list,
            "current_task_index": task_index,
            "current_task_messages": messages,
        }

    def ask_llm(state: CodingAgentState) -> dict:
        response = executor_model.invoke(state["current_task_messages"])
        return {
            "current_task_messages": state["current_task_messages"] + [response],
            "messages": [response],
            "turns": state.get("turns", 0) + 1,
            "tokens": state.get("tokens", 0) + usage_tokens(response),
        }

    def use_tool(state: CodingAgentState) -> dict:
        tool_request = state["current_task_messages"][-1].tool_calls[0]

        tool_name = tool_request["name"]
        tool_args = tool_request["args"]

        tool = TOOLS_BY_NAME[tool_name]
        result = tool.invoke(tool_args)
        response = ToolMessage(content=result, tool_call_id=tool_request["id"])

        update = {
            "current_task_messages": state["current_task_messages"] + [response],
            "messages": [response],
        }

        if tool_name == "verify_script":
            todo_list = list(state["todo_list"])
            current_task = todo_list[state["current_task_index"]]
            current_task.status = ToDoStatus.FAILED if result == "Failed" else ToDoStatus.DONE
            update["todo_list"] = todo_list

        return update

    def end_task(state: CodingAgentState) -> dict:
        todo_list = list(state["todo_list"])
        current_task = todo_list[state["current_task_index"]]
        if current_task.status == ToDoStatus.IN_PROGRESS:
            current_task.status = ToDoStatus.DONE
        outcome = "successfully completed" if current_task.status == ToDoStatus.DONE else "failed"
        return {
            "messages": [AIMessage(content=f"Task {current_task.description} {outcome}.")],
            "todo_list": todo_list,
            "current_task_messages": [],
            "current_task_index": None,
        }

    def should_use_tool(state: CodingAgentState) -> str:
        return "use_tool" if state["current_task_messages"][-1].tool_calls else "end_task"

    executor = StateGraph(CodingAgentState)
    executor.add_node("start_task", start_task)
    executor.add_node("end_task", end_task)
    executor.add_node("ask_llm", ask_llm)
    executor.add_node("use_tool", use_tool)

    executor.add_edge(START, "start_task")
    executor.add_edge("start_task", "ask_llm")
    executor.add_conditional_edges(
        "ask_llm",
        should_use_tool,
        {"use_tool": "use_tool", "end_task": "end_task"},
    )
    executor.add_edge("use_tool", "ask_llm")
    executor.add_edge("end_task", END)

    return executor.compile()
