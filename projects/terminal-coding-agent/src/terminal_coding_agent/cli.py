"""Interactive entry point: a chat REPL over the coding agent graph."""

from __future__ import annotations

import argparse
import tempfile
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from terminal_coding_agent import ui
from terminal_coding_agent.config import ENV_PATH, load_local_env
from terminal_coding_agent.graph import make_graph
from terminal_coding_agent.models import build_models
from terminal_coding_agent.session import run_task_turn
from terminal_coding_agent.telemetry import resolve_model_name


def _session_config(*, worktree: Path, session: str) -> dict[str, Any]:
    """One thread per session; renderers are injected here, not hardwired in graph.py."""
    return {
        "configurable": {
            "worktree": str(worktree),
            "thread_id": session,
            "todo_renderer": ui.TodoPanel(),
            "tool_renderer": ui.ToolLog(),
        }
    }


def _render_task_result(result: dict[str, Any]) -> None:
    messages = result.get("messages") or []
    if messages:
        ui.render_reply(messages[-1].content)
    ui.render_budget(result)


def repl(*, graph: Any, config: dict[str, Any], model_name: str) -> None:
    """Read-eval-print loop. Every turn runs one full task; Ctrl-C at the prompt exits."""
    configurable = config["configurable"]
    # Same instance the graph nodes call, so its in-place region covers their updates.
    todos: ui.TodoPanel = configurable["todo_renderer"]
    ui.banner(
        session=configurable["thread_id"],
        worktree=configurable["worktree"],
        model=model_name,
    )
    while True:
        try:
            text = ui.ask().strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not text:
            continue
        try:
            with todos.region():
                result = run_task_turn(graph=graph, config=config, text=text)
            _render_task_result(result)
        except KeyboardInterrupt:
            ui.render_error("turn cancelled")
        except Exception as exc:  # noqa: BLE001 - one bad turn must not end the session
            ui.render_error(str(exc))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terminal-coding-agent")
    parser.add_argument(
        "--worktree",
        default=None,
        help="worktree the agent may touch; default: a throwaway temp dir",
    )
    parser.add_argument("--session", default=None, help="thread id; default: random per session")
    args = parser.parse_args(argv)

    load_local_env(ENV_PATH)
    session = args.session or f"cli-{uuid.uuid4().hex[:8]}"

    with ExitStack() as stack:
        if args.worktree is None:
            # Throwaway default: nothing the agent writes — including .agent/ state —
            # lands in your repo. It must outlive the whole session, so the stack
            # owns the cleanup instead of a `with` around just this line.
            temp = stack.enter_context(tempfile.TemporaryDirectory(prefix="terminal-coding-agent-"))
            worktree = Path(temp)
        else:
            worktree = Path(args.worktree).resolve()
        config = _session_config(worktree=worktree, session=session)

        repl(
            graph=make_graph(config),
            config=config,
            model_name=resolve_model_name(build_models(config).planner),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
