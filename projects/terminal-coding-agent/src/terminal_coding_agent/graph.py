"""Top-level assembly: Plan -> start_task -> run_agent -> end_task -> [recover] -> Summary."""

from __future__ import annotations

import inspect
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from opentelemetry import trace

from terminal_coding_agent.checkpoint import build_checkpointer
from terminal_coding_agent.config import ENV_PATH, load_local_env
from terminal_coding_agent.demo import DEMO_TASK, seed_demo_worktree
from terminal_coding_agent.executor import build_execute_nodes
from terminal_coding_agent.models import build_models
from terminal_coding_agent.planner import build_planner
from terminal_coding_agent.recover import build_recover
from terminal_coding_agent.state import CodingAgentState, ToDoStatus, format_todos
from terminal_coding_agent.summary import build_summary
from terminal_coding_agent.telemetry import otel_jsonl_path, setup_tracing
from terminal_coding_agent.telemetry.langfuse_callback import (
    flush_langfuse,
    langfuse_callback_handler,
)
from terminal_coding_agent.tools import make_tools

# projects/terminal-coding-agent (not the monorepo root)
_AGENT_PROJECT_DIR = Path(__file__).resolve().parents[2]


def _after_make_plan(state: CodingAgentState) -> str:
    if state.get("stop_reason") or not state.get("todo_list"):
        return "summary"
    return "start_task"


def _after_start(state: CodingAgentState) -> str:
    if state.get("stop_reason") or state.get("current_task_index") is None:
        return "summary"
    return "run_agent"


def _after_end_task(state: CodingAgentState) -> str:
    if state.get("stop_reason"):
        return "summary"
    if state.get("blocked_reason"):
        return "recover"
    todo_list = state.get("todo_list") or []
    settled = {ToDoStatus.DONE, ToDoStatus.DEPRECATED}
    if not todo_list or all(item.status in settled for item in todo_list):
        return "summary"
    return "start_task"


def _after_recover(state: CodingAgentState) -> str:
    return "summary" if state.get("stop_reason") else "start_task"


def _announce_todos(node: Callable) -> Callable:
    """Print the todo list a node just rewrote — the graph's only console output.

    The print lives here rather than in the nodes because `replace_todos` runs
    twice per write (once for the conditional-edge read, once for apply_writes),
    so a print inside the reducer repeated itself; the assembly layer sees each
    update exactly once. Nodes that do not rewrite `todo_list` stay silent.
    """
    takes_config = len(inspect.signature(node).parameters) > 1

    def announced(state: CodingAgentState, config: RunnableConfig) -> dict[str, Any]:
        updates = node(state, config) if takes_config else node(state)
        if todo_list := updates.get("todo_list"):
            version = updates.get("replan_count", state.get("replan_count", 0))
            print(format_todos(todo_list, version))
        return updates

    return announced


def _write_mermaid_png(compiled: CompiledStateGraph, config: RunnableConfig) -> None:
    """Write a PNG when mermaid_path is set. Nothing is printed or opened."""
    raw = (config.get("configurable") or {}).get("mermaid_path")
    if not raw:
        return
    png_path = Path(raw)
    if png_path.suffix.lower() != ".png":
        png_path = png_path / "graph.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.unlink(missing_ok=True)
    png_path.write_bytes(compiled.get_graph(xray=True).draw_mermaid_png())


def make_graph(config: RunnableConfig) -> CompiledStateGraph:
    """Assemble the top-level graph with IN_PROGRESS committed before the agent runs."""
    models = build_models(config)

    configurable = config.get("configurable") or {}
    raw = configurable.get("worktree")
    worktree = Path(raw).resolve() if raw else Path.cwd().resolve()
    sequence_raw = configurable.get("sequence_path")
    sequence_path = Path(sequence_raw) if sequence_raw else None
    sequence_events: list | None = [] if sequence_path else None

    setup_tracing(worktree=worktree)

    # Plain function node (not a nested StateGraph) so replace_todos does not fire twice.
    make_plan = build_planner(models)
    execute = build_execute_nodes(
        models,
        make_tools(worktree),
        worktree=worktree,
        sequence_events=sequence_events,
    )
    recover = build_recover(make_plan, worktree=worktree)

    coding_agent = StateGraph(CodingAgentState)

    summary = build_summary(
        models,
        worktree=worktree,
        sequence_events=sequence_events,
        sequence_path=sequence_path,
    )

    coding_agent.add_node("make_plan", _announce_todos(make_plan))
    coding_agent.add_node("start_task", _announce_todos(execute["start_task"]))
    coding_agent.add_node("run_agent", execute["run_agent"])
    coding_agent.add_node("end_task", _announce_todos(execute["end_task"]))
    coding_agent.add_node("recover", _announce_todos(recover))
    coding_agent.add_node("summary", summary)

    coding_agent.add_edge(START, "make_plan")
    coding_agent.add_conditional_edges(
        "make_plan",
        _after_make_plan,
        {"start_task": "start_task", "summary": "summary"},
    )
    coding_agent.add_conditional_edges(
        "start_task",
        _after_start,
        {"run_agent": "run_agent", "summary": "summary"},
    )
    coding_agent.add_edge("run_agent", "end_task")
    coding_agent.add_conditional_edges(
        "end_task",
        _after_end_task,
        {"start_task": "start_task", "recover": "recover", "summary": "summary"},
    )
    coding_agent.add_conditional_edges(
        "recover",
        _after_recover,
        {"start_task": "start_task", "summary": "summary"},
    )
    coding_agent.add_edge("summary", END)

    compiled = coding_agent.compile(checkpointer=build_checkpointer(worktree))
    _write_mermaid_png(compiled, config)

    thread_id = str(configurable.get("thread_id") or "default")
    bound_config: dict = {
        "configurable": {"thread_id": thread_id},
        "run_name": "terminal-coding-agent",
    }
    handler = langfuse_callback_handler()
    if handler is not None:
        # Framework integration: callbacks propagate into nested create_agent / model invokes.
        bound_config["callbacks"] = [handler]
    return compiled.with_config(bound_config)


if __name__ == "__main__":
    # Env must load before Langfuse client init (skill: import order).
    load_local_env(ENV_PATH)
    with tempfile.TemporaryDirectory() as temp_dir:
        worktree = Path(temp_dir)
        seed_demo_worktree(worktree)
        agent = make_graph(
            {
                "configurable": {
                    "worktree": str(worktree),
                    "mermaid_path": str(Path(__file__).resolve().parents[2] / "graph.png"),
                    "sequence_path": str(Path(__file__).resolve().parents[2] / "sequence.png"),
                }
            }
        )

        from langfuse import propagate_attributes

        with propagate_attributes(
            trace_name="terminal-coding-agent-demo",
            tags=["terminal-coding-agent", "demo", "langgraph"],
            metadata={"framework": "langgraph", "role": "demo"},
        ):
            response = agent.invoke(
                {"messages": [HumanMessage(content=DEMO_TASK)]},
                config={"configurable": {"thread_id": "demo"}},
            )

        for message in response["messages"]:
            message.pretty_print()
        print("------ Budget --------")
        print(f"turns:             {response.get('turns', 0)}")
        print(f"tokens:            {response.get('tokens', 0)}")
        print(f"input_tokens:      {response.get('input_tokens', 0)}")
        print(f"output_tokens:     {response.get('output_tokens', 0)}")
        print(f"cache_read_tokens: {response.get('cache_read_tokens', 0)}")
        print(f"cost_rmb:          {response.get('cost_rmb', 0.0):.6f}")
        print(f"stop_reason:       {response.get('stop_reason')}")
        print(f"replan_count:      {response.get('replan_count', 0)}")
        print("----------------------")
        print(format_todos(response.get("todo_list") or [], response.get("replan_count", 0)))

        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
        flush_langfuse()

        src = otel_jsonl_path(worktree)
        dst = Path(str(_AGENT_PROJECT_DIR).rstrip("/") + "/otel.jsonl")
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"otel spans copied to {dst}")
        else:
            print(f"otel span file missing: {src}")
