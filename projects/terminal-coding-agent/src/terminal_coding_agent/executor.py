"""Execute stage: act / observe / recover subgraph that runs todos one by one."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from terminal_coding_agent.models import SYSTEM_PROMPTS, AgentModels
from terminal_coding_agent.state import CodingAgentState, ToDoStatus, usage_tokens


def build_executor(models: AgentModels, tools: list) -> CompiledStateGraph:
    """Build the execution subgraph: start_task -> (ask_llm <-> use_tool)* -> end_task."""
    tools_map = {t.name: t for t in tools}
    executor_model = models.executor.bind_tools(tools)

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
        last = state["current_task_messages"][-1]
        tool_messages: list[ToolMessage] = []
        for tool_request in last.tool_calls:
            tool_name = tool_request["name"]
            tool_args = tool_request["args"]
            tool = tools_map.get(tool_name)
            if tool is None:
                result = f"Error: unknown tool: {tool_name}"
            else:
                result = tool.invoke(tool_args)
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_request["id"])
            )

        return {
            "current_task_messages": state["current_task_messages"] + tool_messages,
            "messages": tool_messages,
        }

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
