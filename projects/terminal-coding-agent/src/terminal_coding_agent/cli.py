"""Interactive entry point: a chat REPL over the coding agent graph."""

import argparse
import tempfile
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, ExitStack, nullcontext
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from terminal_coding_agent import commands, shorthands, ui
from terminal_coding_agent.config import ENV_PATH, SHELL_TIMEOUT_SECONDS, load_local_env
from terminal_coding_agent.graph import make_graph
from terminal_coding_agent.models import build_models
from terminal_coding_agent.session import pending_interrupts, resume_turn, run_task_turn
from terminal_coding_agent.telemetry import resolve_model_name
from terminal_coding_agent.tools.shell import _run


def _run_shell_in_worktree(command: str, worktree: Path) -> tuple[int | None, str]:
    """Run one `!` command in the worktree. `None` exit code means it timed out."""
    result = _run(command, worktree, SHELL_TIMEOUT_SECONDS)
    if result is None:
        return None, f"timed out after {SHELL_TIMEOUT_SECONDS}s"
    exit_code, stdout, stderr = result
    return exit_code, stdout + stderr


def _session_config(*, worktree: Path, session: str) -> dict[str, Any]:
    """One thread per session; renderers are injected here, not hardwired in graph.py."""
    todos = ui.TodoPanel()
    configurable: dict[str, Any] = {
        "worktree": str(worktree),
        "thread_id": session,
        "todo_renderer": todos,
        "tool_renderer": ui.ToolLog(),
        "enable_hitl": True,
        "enable_answer_mode": True,
        # "local_model": "qwen3:8b"
    }
    if todos.redraws_in_place:
        configurable["summary_renderer"] = todos.stream_summary
    return {"configurable": configurable}


def _render_task_result(result: dict[str, Any], *, show_reply: bool = True) -> None:
    messages = result.get("messages") or []
    if messages and show_reply:
        ui.render_reply(messages[-1].content)
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


def repl(*, graph: Any, config: dict[str, Any], model_name: str) -> None:
    """Read-eval-print loop. Every turn runs one full task; Ctrl-C at the prompt exits."""
    configurable = config["configurable"]
    worktree = Path(configurable["worktree"])
    session = ui.make_session(worktree=worktree, command_names=tuple(commands.COMMANDS))
    # Same instance the graph nodes call, so its in-place region covers their updates.
    todos: ui.TodoPanel = configurable["todo_renderer"]
    streamed_reply = callable(configurable.get("summary_renderer"))
    ui.banner(
        session=configurable["thread_id"],
        worktree=configurable["worktree"],
        model=model_name,
    )
    # A killed process leaves the approval in the checkpoint; rebuild it before reading input.
    if pending := pending_interrupts(graph, config):
        try:
            result = _drive_approvals(
                graph=graph, config=config, pending=pending[0], region=todos.region
            )
            _render_task_result(result, show_reply=not streamed_reply)
        except _StdinClosedError:
            return
        except KeyboardInterrupt:
            ui.render_error("turn cancelled")
        except Exception as exc:  # noqa: BLE001 - a bad resume must not end the session
            ui.render_error(str(exc))
    while True:
        try:
            text = ui.ask(session=session).strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not text:
            continue
        try:
            parsed = shorthands.parse(text)
            if parsed.kind == "shell":
                exit_code, output = _run_shell_in_worktree(parsed.text, worktree)
                ui.render_shell(exit_code=exit_code, output=output)
                continue
            if parsed.kind == "command":
                outcome = commands.dispatch(
                    parsed.command, parsed.text, commands.CommandContext(worktree=worktree)
                )
                if outcome is None:
                    ui.render_error(f"unknown command: /{parsed.command}")
                    continue
                if outcome.message is not None:
                    ui.render_local(outcome.message)
                if outcome.quit:
                    return
                if outcome.prompt is None:
                    continue
                message = HumanMessage(content=outcome.prompt)
            else:
                message = shorthands.build_task_message(parsed, worktree)
        except shorthands.AttachmentError as exc:
            ui.render_error(str(exc))
            continue
        try:
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
            _render_task_result(result, show_reply=not streamed_reply)
        except _StdinClosedError:
            # EOF during a question: nothing else can be read, so end the session
            # instead of re-entering the drain loop with no answer.
            return
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
        for command in commands.builtin_commands():
            commands.register(command)

        repl(
            graph=make_graph(config),
            config=config,
            model_name=resolve_model_name(build_models(config).planner),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
