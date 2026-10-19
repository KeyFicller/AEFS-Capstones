"""Top-level assembly: [intent] -> run_agent -> summary.

One agent owns the whole request; `TodoListMiddleware` (inside the executor) is the
planner, so no plan node and no approval gate exist above it.
"""

import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
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
from terminal_coding_agent.state import CodingAgentState, format_todos
from terminal_coding_agent.summary import build_summary
from terminal_coding_agent.tools import make_tools
from terminal_coding_agent.viz import compose_mermaid

# projects/terminal-coding-agent (not the monorepo root)
_AGENT_PROJECT_DIR = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def _after_intent(state: CodingAgentState) -> str:
    return "chat" if state.get("intent") == "chat" else "run_agent"


def _stdout(text: str) -> None:
    """Harbor and the module entry point read these lines from stdout."""
    print(text)  # noqa: T201


def _print_todos(text: str) -> None:
    """Harbor's fallback renderer: plain text, no chrome, straight into the log."""
    _stdout(text)


def todos_renderer(config: RunnableConfig | None) -> Callable[[str], None]:
    """Console sink for the todo panel: config-injected, else plain print (Harbor default)."""
    configurable = (config or {}).get("configurable") or {}
    render = configurable.get("todo_renderer")
    return render if callable(render) else _print_todos


def _write_mermaid_png(
    compiled: CompiledStateGraph, config: RunnableConfig, inner: Any | None = None
) -> None:
    """Write a PNG when mermaid_path is set. Nothing is printed or opened.

    `inner` is the executor's own compiled agent: the executor node builds its agent at
    runtime, so `xray=True` cannot see into it and its graph is pasted into the node box.

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
        mermaid = compiled.get_graph(xray=True).draw_mermaid()
        if inner is not None:
            mermaid = compose_mermaid(
                mermaid, inner.get_graph(xray=True).draw_mermaid(), "run_agent"
            )
        png_bytes = draw_mermaid_png(mermaid)
    except Exception:  # noqa: BLE001 - a missing diagram must not break the graph
        logger.warning("could not render %s", png_path, exc_info=True)
        return
    png_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.write_bytes(png_bytes)


def make_graph(config: RunnableConfig) -> CompiledStateGraph:
    """Assemble the top-level graph: one agent run, then the Stop hook."""
    models = build_models(config)

    configurable = config.get("configurable") or {}
    raw = configurable.get("worktree")
    worktree = Path(raw).resolve() if raw else Path.cwd().resolve()
    sequence_raw = configurable.get("sequence_path")
    sequence_path = Path(sequence_raw) if sequence_raw else None
    sequence_events: list | None = [] if sequence_path else None

    # Run state (checkpoint / trace / otel / patch) lives outside the worktree when the
    # caller supplies a directory; `--repo` does, so a real repository stays clean.
    raw_artifacts = configurable.get("artifacts_dir")
    artifacts = Path(str(raw_artifacts)) if raw_artifacts else None
    if artifacts is not None:
        artifacts.mkdir(parents=True, exist_ok=True)

    setup_tracing(worktree=worktree, artifacts=artifacts)

    enable_intent = bool(configurable.get("enable_intent"))
    enable_debug = bool(configurable.get("enable_debug"))
    shell_anything = bool(configurable.get("shell_anything"))
    allow_dangerous = bool(configurable.get("allow_dangerous"))
    execute = build_execute_nodes(
        models,
        make_tools(
            worktree,
            enable_web_search=bool(configurable.get("enable_web_search")),
            enable_debug=enable_debug,
            shell_anything=shell_anything,
            sandbox=configurable.get("sandbox"),
            allow_dangerous=allow_dangerous,
        ),
        sequence_events=sequence_events,
        enable_debug=enable_debug,
        enable_hitl=bool(configurable.get("enable_hitl")),
        shell_anything=shell_anything,
        allow_dangerous=allow_dangerous,
        executor_only=bool(configurable.get("executor_only")),
        todo_render=todos_renderer(config),
    )

    coding_agent = StateGraph(CodingAgentState)
    summary = build_summary(
        models,
        worktree=worktree,
        artifacts=artifacts,
        sequence_events=sequence_events,
        sequence_path=sequence_path,
    )

    def _finish(graph: StateGraph) -> CompiledStateGraph:
        compiled = graph.compile(checkpointer=build_checkpointer(worktree, artifacts=artifacts))
        # Only pay for the second compile when a diagram was actually asked for.
        inner = execute["diagram_agent"]() if configurable.get("mermaid_path") else None
        _write_mermaid_png(compiled, config, inner)

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

    coding_agent.add_node("run_agent", execute["run_agent"])
    coding_agent.add_node("summary", summary)
    if enable_intent:
        coding_agent.add_node("intent", build_intent(models))
        coding_agent.add_node("chat", execute["chat"])
        coding_agent.add_edge("chat", "summary")
        coding_agent.add_edge(START, "intent")
        coding_agent.add_conditional_edges(
            "intent", _after_intent, {"chat": "chat", "run_agent": "run_agent"}
        )
    else:
        coding_agent.add_edge(START, "run_agent")
    coding_agent.add_edge("run_agent", "summary")
    coding_agent.add_edge("summary", END)

    return _finish(coding_agent)


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
                    "enable_intent": os.environ.get("ENABLE_INTENT", "1") != "0",
                    # The CLI topology includes ask_user; Harbor runs without it.
                    "enable_hitl": os.environ.get("ENABLE_HITL", "1") != "0",
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
        _stdout(f"stop_reason:       {response.get('stop_reason')}")
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
