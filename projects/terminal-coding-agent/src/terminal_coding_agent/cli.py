"""Interactive entry point: a chat REPL over the coding agent graph."""

import argparse
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, nullcontext
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from repl_console import EndSessionError, Repl, render_error, render_reply
from telemetry import resolve_model_name

from terminal_coding_agent import ui
from terminal_coding_agent.config import (
    ENV_PATH,
    SHELL_TIMEOUT_SECONDS,
    TOOL_OUTPUT_TOKEN_LIMIT,
    load_local_env,
)
from terminal_coding_agent.graph import make_graph
from terminal_coding_agent.models import build_models
from terminal_coding_agent.session import pending_interrupts, resume_turn, run_task_turn


def _session_config(*, worktree: Path, session: str) -> dict[str, Any]:
    """One thread per session; renderers are injected here, not hardwired in graph.py."""
    return {
        "configurable": {
            "worktree": str(worktree),
            "thread_id": session,
            "todo_renderer": ui.TodoPanel(),
            "tool_renderer": ui.ToolLog(),
            "enable_hitl": True,
            "enable_intent": True,
            "enable_web_search": True,
            "enable_debug": True,
            # "local_model": "qwen3:8b"
        }
    }


def _render_task_result(result: dict[str, Any]) -> None:
    messages = result.get("messages") or []
    if messages:
        render_reply(messages[-1].content)
    ui.render_budget(result)


class _StdinClosedError(Exception):
    """`input()` saw EOF while a question was pending: no answer can arrive again."""


def _answer_prompt(payload: Any, show_plan: bool) -> Any:
    """Resume value for one pending interrupt, dispatched on the payload's shape.

    A plan gate resumes with a string decision; an `ask_user` question resumes with the
    answer itself. Dismissal means different things to each, so it is mapped here.
    """
    if isinstance(payload, Mapping) and payload.get("type") == "question":
        try:
            return ui.ask_question(payload)
        except EOFError:
            # Ctrl-D (piped stdin, Harbor log) cannot answer the next question either;
            # resuming with "carry on" would let the model ask again and loop forever.
            raise _StdinClosedError from None
        except KeyboardInterrupt:
            # A dismissed question is not a rejection: the agent carries on by itself.
            return {"answer": None, "cancelled": True}
    try:
        return ui.ask_approval(payload, show_plan=show_plan)
    except (EOFError, KeyboardInterrupt):
        # A declined plan prompt still walks the Stop hook, instead of leaving a live pause.
        return "reject"


def _drive_approvals(
    *,
    graph: Any,
    config: dict[str, Any],
    pending: Any,
    show_plan: bool = True,
    region: Callable[[], AbstractContextManager[Any]] | None = None,
) -> dict[str, Any]:
    """Answer pending approvals in-turn until the graph stops pausing.

    `show_plan` applies to the first prompt only; a pause after a resume shows its own plan.
    `region` wraps the resumed graph, not the prompt — `input()` must stay out of the Live's way.
    """
    open_region = region or nullcontext
    while True:
        decision = _answer_prompt(pending.value, show_plan)
        show_plan = True

        with open_region():
            result = resume_turn(graph=graph, config=config, decision=decision)
        remaining = result.get("__interrupt__") or []
        if not remaining:
            return result
        pending = remaining[0]


def repl(
    *,
    graph: Any,
    config: dict[str, Any],
    model_name: str,
    commands: Sequence[Any] = (),
) -> None:
    """Read-eval-print loop. Every turn runs one full task; Ctrl-C at the prompt exits."""
    configurable = config["configurable"]
    worktree = Path(configurable["worktree"])
    # Same instance the graph nodes call, so its in-place region covers their updates.
    todos: ui.TodoPanel = configurable["todo_renderer"]

    def on_ready() -> None:
        if pending := pending_interrupts(graph, config):
            try:
                result = _drive_approvals(
                    graph=graph, config=config, pending=pending[0], region=todos.region
                )
                _render_task_result(result)
            except _StdinClosedError as exc:
                raise EndSessionError from exc
            except KeyboardInterrupt:
                render_error("turn cancelled")
            except Exception as exc:  # noqa: BLE001 - a bad resume must not end the session
                render_error(str(exc))

    def on_task(message: HumanMessage) -> None:
        with todos.region():
            result = run_task_turn(graph=graph, config=config, message=message)
        if pending := result.get("__interrupt__"):
            result = _drive_approvals(
                graph=graph,
                config=config,
                pending=pending[0],
                show_plan=False,
                region=todos.region,
            )
        _render_task_result(result)

    def guarded(message: HumanMessage) -> None:
        try:
            on_task(message)
        except _StdinClosedError as exc:
            raise EndSessionError from exc
        except KeyboardInterrupt:
            render_error("turn cancelled")
        except EndSessionError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad turn must not end the session
            render_error(str(exc))

    Repl(
        title="terminal-coding-agent",
        info={
            "session": str(configurable["thread_id"]),
            "worktree": str(configurable["worktree"]),
            "model": model_name,
        },
        root=worktree,
        on_task=guarded,
        commands=commands,
        shell_timeout=SHELL_TIMEOUT_SECONDS,
        text_token_limit=TOOL_OUTPUT_TOKEN_LIMIT,
        on_ready=on_ready,
    ).run()


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
