"""Top-level assembly: make_plan -> start_task -> run_agent -> end_task -> [recover] -> summary.

With `enable_hitl`, both entry points (`make_plan`, `recover`) route through `await_plan_approval`.
"""

import inspect
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from opentelemetry import trace
from telemetry import otel_jsonl_path, setup_tracing
from telemetry.langfuse_callback import (
    flush_langfuse,
    langfuse_callback_handler,
)

from terminal_coding_agent.checkpoint import build_checkpointer
from terminal_coding_agent.config import ENV_PATH, load_local_env
from terminal_coding_agent.demo import DEMO_TASK, seed_demo_worktree
from terminal_coding_agent.executor import build_execute_nodes
from terminal_coding_agent.gate import build_intent
from terminal_coding_agent.models import build_models
from terminal_coding_agent.planner import build_planner
from terminal_coding_agent.recover import build_recover
from terminal_coding_agent.state import CodingAgentState, ToDoStatus, format_todos
from terminal_coding_agent.summary import build_summary
from terminal_coding_agent.tools import make_tools

# projects/terminal-coding-agent (not the monorepo root)
_AGENT_PROJECT_DIR = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def _after_make_plan(state: CodingAgentState) -> str:
    if state.get("stop_reason"):
        return "summary"
    if not state.get("todo_list"):
        return "summary"
    return "start_task"


def _after_make_plan_gated(state: CodingAgentState) -> str:
    if state.get("stop_reason"):
        return "summary"
    if not state.get("todo_list"):
        return "summary"
    return "await_plan_approval"


def _await_plan_approval(state: CodingAgentState) -> dict[str, Any]:
    """Plan gate. Nothing runs before `interrupt()`: the node re-runs from the top on resume."""
    decision = interrupt({"plan": format_todos(state.get("todo_list") or [])})
    if decision == "approve":
        return {}

    # Non-zero means `recover` rewrote the plan: a different failure mode, counted separately.
    if state.get("replan_count"):
        return {"stop_reason": "replan_rejected"}
    return {"stop_reason": "plan_rejected"}


def _after_plan_approval(state: CodingAgentState) -> str:
    if state.get("stop_reason"):
        return "summary"
    return "start_task"


def _after_intent(state: CodingAgentState) -> str:
    return "chat" if state.get("intent") == "chat" else "work"


def _after_start(state: CodingAgentState) -> str:
    if state.get("stop_reason") or state.get("current_task_index") is None:
        return "summary"
    return "run_agent"


_SETTLED_STATUSES = frozenset({ToDoStatus.DONE, ToDoStatus.DEPRECATED})


def _has_unsettled_work(state: CodingAgentState) -> bool:
    """True while some todo still needs the agent: neither finished nor retired."""
    return any(item.status not in _SETTLED_STATUSES for item in state.get("todo_list") or [])


def _after_end_task(state: CodingAgentState) -> str:
    if state.get("stop_reason"):
        return "summary"
    if state.get("blocked_reason"):
        return "recover"
    if not _has_unsettled_work(state):
        return "summary"
    return "start_task"


def _after_recover(state: CodingAgentState) -> str:
    # `start_task` asserts on a missing PENDING item, so the same guard applies here.
    if state.get("stop_reason") or not _has_unsettled_work(state):
        return "summary"
    return "start_task"


def _after_recover_gated(state: CodingAgentState) -> str:
    """`_after_recover` with the gate in place of the bare hand-off to `start_task`."""
    if state.get("stop_reason") or not _has_unsettled_work(state):
        return "summary"
    return "await_plan_approval"


def _stdout(text: str) -> None:
    """Harbor and the module entry point read these lines from stdout."""
    print(text)  # noqa: T201


def _print_todos(text: str, version: int = 0) -> None:
    """Harbor's fallback renderer: plain text, no chrome, straight into the log."""
    _stdout(f"replan v{version}")
    _stdout(text)


def todos_renderer(config: RunnableConfig | None) -> Callable[[str, int], None]:
    """Console sink for the todo panel: config-injected, else plain print (Harbor default)."""
    configurable = (config or {}).get("configurable") or {}
    render = configurable.get("todo_renderer")
    return render if callable(render) else _print_todos


def _announce_todos(node: Callable) -> Callable:
    """Print the todo list a node just rewrote — the graph's only console output.

    The print lives here rather than in the nodes because `replace_todos` runs
    twice per write (once for the conditional-edge read, once for apply_writes),
    so a print inside the reducer repeated itself; the assembly layer sees each
    update exactly once. Nodes that do not rewrite `todo_list` stay silent.
    """
    takes_config = "config" in inspect.signature(node).parameters

    def announced(state: CodingAgentState, config: RunnableConfig) -> dict[str, Any]:
        updates = node(state, config) if takes_config else node(state)
        if todo_list := updates.get("todo_list"):
            version = updates.get("replan_count", state.get("replan_count", 0))
            todos_renderer(config)(format_todos(todo_list), version)
        return updates

    return announced


def _write_mermaid_png(compiled: CompiledStateGraph, config: RunnableConfig) -> None:
    """Write a PNG when mermaid_path is set. Nothing is printed or opened.

    Rendering goes through mermaid.ink, so it is unavailable in an isolated container
    and must not be able to fail `make_graph`. Bytes are rendered before touching the
    target, leaving any previous diagram in place when rendering fails.
    """
    raw = (config.get("configurable") or {}).get("mermaid_path")
    if not raw:
        return
    png_path = Path(raw)
    if png_path.suffix.lower() != ".png":
        png_path = png_path / "graph.png"
    try:
        png_bytes = compiled.get_graph(xray=True).draw_mermaid_png()
    except Exception:  # noqa: BLE001 - a missing diagram must not break the graph
        logger.warning("could not render %s", png_path, exc_info=True)
        return
    png_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.write_bytes(png_bytes)


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
    enable_intent = bool(configurable.get("enable_intent"))
    make_plan = build_planner(models, with_system_prompt=enable_intent)
    enable_debug = bool(configurable.get("enable_debug"))
    execute = build_execute_nodes(
        models,
        make_tools(
            worktree,
            enable_web_search=bool(configurable.get("enable_web_search")),
            enable_debug=enable_debug,
        ),
        worktree=worktree,
        sequence_events=sequence_events,
        enable_debug=enable_debug,
        enable_hitl=bool(configurable.get("enable_hitl")),
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

    plan_targets = {"start_task": "start_task", "summary": "summary"}

    if enable_intent:
        coding_agent.add_node("intent", build_intent(models))
        coding_agent.add_node("chat", execute["chat"])
        coding_agent.add_edge("chat", "summary")
        coding_agent.add_edge(START, "intent")
        coding_agent.add_conditional_edges(
            "intent", _after_intent, {"chat": "chat", "work": "make_plan"}
        )
    else:
        coding_agent.add_edge(START, "make_plan")
    if configurable.get("enable_hitl"):
        coding_agent.add_node("await_plan_approval", _await_plan_approval)
        coding_agent.add_conditional_edges(
            "make_plan",
            _after_make_plan_gated,
            {**plan_targets, "await_plan_approval": "await_plan_approval"},
        )
        coding_agent.add_conditional_edges(
            "await_plan_approval",
            _after_plan_approval,
            {"start_task": "start_task", "summary": "summary"},
        )
    else:
        coding_agent.add_conditional_edges("make_plan", _after_make_plan, plan_targets)
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
    if configurable.get("enable_hitl"):
        coding_agent.add_conditional_edges(
            "recover",
            _after_recover_gated,
            {"await_plan_approval": "await_plan_approval", "summary": "summary"},
        )
    else:
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
                    # Both gates on so graph.png shows the CLI topology; 0 renders Harbor's.
                    "enable_hitl": os.environ.get("ENABLE_HITL", "1") != "0",
                    "enable_intent": os.environ.get("ENABLE_INTENT", "1") != "0",
                }
            }
        )

        from langfuse import propagate_attributes

        demo_config = {"configurable": {"thread_id": "demo"}}
        with propagate_attributes(
            trace_name="terminal-coding-agent-demo",
            tags=["terminal-coding-agent", "demo", "langgraph"],
            metadata={"framework": "langgraph", "role": "demo"},
        ):
            response = agent.invoke(
                {"messages": [HumanMessage(content=DEMO_TASK)]}, config=demo_config
            )
            # Non-interactive: approve every gate, else the run stops here with an empty transcript.
            while response.get("__interrupt__"):
                response = agent.invoke(Command(resume="approve"), config=demo_config)

        for message in response["messages"]:
            message.pretty_print()
        _stdout("------ Budget --------")
        _stdout(f"turns:             {response.get('turns', 0)}")
        _stdout(f"tokens:            {response.get('tokens', 0)}")
        _stdout(f"input_tokens:      {response.get('input_tokens', 0)}")
        _stdout(f"output_tokens:     {response.get('output_tokens', 0)}")
        _stdout(f"cache_read_tokens: {response.get('cache_read_tokens', 0)}")
        _stdout(f"cost_rmb:          {response.get('cost_rmb', 0.0):.6f}")
        _stdout(f"stop_reason:       {response.get('stop_reason')}")
        _stdout(f"replan_count:      {response.get('replan_count', 0)}")
        _stdout("----------------------")
        _stdout(format_todos(response.get("todo_list") or []))

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
            _stdout(f"otel spans copied to {dst}")
        else:
            _stdout(f"otel span file missing: {src}")
