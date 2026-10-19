"""Interactive entry point: a chat REPL over the coding agent graph."""

import argparse
import os
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from repl_console import EndSessionError, Repl, render_error, render_reply
from repl_console.commands import CommandContext, CommandOutcome, LocalCommand
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
from terminal_coding_agent.sandbox import DockerSandbox, SandboxError, require_docker
from terminal_coding_agent.session import pending_interrupts, resume_turn, run_task_turn
from terminal_coding_agent.tools import display

WORKSPACES_DIRNAME = ".agent-workspaces"
CACHE_DIRNAME = "terminal-coding-agent"
GIT_TIMEOUT_SECONDS = 15


class RepoError(RuntimeError):
    """`--repo` did not point at a usable git repository."""


def _git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """One read-only git query; a missing git binary is reported like a failed query."""
    try:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RepoError(f"git is not available: {exc}") from exc


def repo_root(raw: str) -> Path:
    """Top level of the git repository containing `raw`; a subdirectory resolves upward."""
    result = _git(Path(raw), "rev-parse", "--show-toplevel")
    if result.returncode != 0 or not result.stdout.strip():
        detail = result.stderr.strip() or "not a git repository"
        raise RepoError(f"{raw}: {detail}")
    return Path(result.stdout.strip()).resolve()


def has_uncommitted_changes(repo: Path) -> bool:
    """True when `git diff HEAD` would mix the user's edits into this run's patch."""
    return bool(_git(repo, "status", "--porcelain").stdout.strip())


def artifacts_dir(session: str) -> Path:
    """Run state for a `--repo` session, kept outside the repository."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    path = cache / CACHE_DIRNAME / session
    path.mkdir(parents=True, exist_ok=True)
    return path


class ToolOutputCommand(LocalCommand):
    """`/tool_output off|simple|detail`: pick how much of each tool call prints.

    The level is the same module-level switch `main` sets, so `off` matches what a
    Harbor run looks like. It only takes effect at the prompt, hence from the next
    tool call.
    """

    name = "tool_output"
    summary = "tool-call output: off (silent), simple (one line), detail (full)"

    def run(self, ctx: CommandContext, args: str) -> CommandOutcome:  # noqa: ARG002
        choice = args.strip().lower()
        usage = "   usage: /tool_output off|simple|detail"
        if choice not in display.LEVELS:
            return CommandOutcome(message=f"tool output {display.LEVEL}{usage}")
        display.LEVEL = choice
        return CommandOutcome(message=f"tool output {choice}")


def _session_config(
    *,
    worktree: Path,
    session: str,
    shell_anything: bool = False,
    allow_dangerous: bool = False,
    sandbox: DockerSandbox | None = None,
    artifacts: Path | None = None,
) -> dict[str, Any]:
    """One thread per session; the todo renderer is injected here, not in graph.py."""
    configurable: dict[str, Any] = {
        "worktree": str(worktree),
        "thread_id": session,
        "todo_renderer": ui.TodoPanel(),
        "enable_hitl": False,
        "enable_intent": True,
        "enable_web_search": True,
        "enable_debug": True,
        "shell_anything": shell_anything,
        "allow_dangerous": allow_dangerous,
        "sandbox": sandbox,
        # "local_model": "qwen3:8b"
    }
    if artifacts is not None:
        configurable["artifacts_dir"] = str(artifacts)
    return {"configurable": configurable}


def _render_task_result(result: dict[str, Any]) -> None:
    messages = result.get("messages") or []
    if messages:
        render_reply(messages[-1].content)
    ui.render_budget(result)


class _StdinClosedError(Exception):
    """`input()` saw EOF while a question was pending: no answer can arrive again."""


def _answer_prompt(payload: Any) -> Any:
    """Resume value for the pending `ask_user` interrupt: the answer itself."""
    if not (isinstance(payload, Mapping) and payload.get("type") == "question"):
        raise ValueError(f"unknown interrupt payload: {payload!r}")
    try:
        return ui.ask_question(payload)
    except EOFError:
        # Ctrl-D (piped stdin, Harbor log) cannot answer the next question either;
        # resuming with "carry on" would let the model ask again and loop forever.
        raise _StdinClosedError from None
    except KeyboardInterrupt:
        # A dismissed question is not a rejection: the agent carries on by itself.
        return {"answer": None, "cancelled": True}


def _drive_approvals(
    *,
    graph: Any,
    config: dict[str, Any],
    pending: Any,
    region: Callable[[], AbstractContextManager[Any]] | None = None,
) -> dict[str, Any]:
    """Answer pending questions in-turn until the graph stops pausing.

    `region` wraps the resumed graph, not the prompt — `input()` must stay out of the Live's way.
    """
    open_region = region or nullcontext
    while True:
        decision = _answer_prompt(pending.value)

        with open_region():
            result = resume_turn(graph=graph, config=config, decision=decision)
        remaining = result.get("__interrupt__") or []
        if not remaining:
            return result
        pending = remaining[0]


def _repl_info(configurable: Mapping[str, Any], *, model_name: str) -> dict[str, str]:
    """Header lines; `artifacts` only shows up when run state lives outside the worktree."""
    info = {
        "session": str(configurable["thread_id"]),
        "worktree": str(configurable["worktree"]),
        "model": model_name,
    }
    if artifacts := configurable.get("artifacts_dir"):
        info["artifacts"] = str(artifacts)
    return info


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
        info=_repl_info(configurable, model_name=model_name),
        root=worktree,
        on_task=guarded,
        commands=commands,
        shell_timeout=SHELL_TIMEOUT_SECONDS,
        text_token_limit=TOOL_OUTPUT_TOKEN_LIMIT,
        on_ready=on_ready,
    ).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terminal-coding-agent")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--worktree",
        default=None,
        help=("worktree the agent may touch; default: ./.agent-workspaces/<session>, kept"),
    )
    source.add_argument(
        "--repo",
        default=None,
        help=(
            "existing git repository to work in, editing it in place; run state "
            "(checkpoint / trace / otel / patch) goes to the cache dir"
        ),
    )
    parser.add_argument("--session", default=None, help="thread id; default: random per session")
    parser.add_argument(
        "--shell_anything",
        action="store_true",
        help="bind only run_shell; read, edit, and git go through the shell",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="run run_shell in a Docker container mounted at the worktree",
    )
    parser.add_argument(
        "--docker-image",
        default="python:3.13-slim",
        help="image for --docker; it must provide the commands the tasks run",
    )
    parser.add_argument(
        "--allow-dangerous",
        action="store_true",
        help=("with --docker: drop the network and command fences, and allow every git subcommand"),
    )
    args = parser.parse_args(argv)

    if args.allow_dangerous and not args.docker:
        render_error(
            "--allow-dangerous requires --docker: without a sandbox the fence is all there is"
        )
        return 2

    load_local_env(ENV_PATH)
    session = args.session or f"cli-{uuid.uuid4().hex[:8]}"

    artifacts: Path | None = None
    if args.repo:
        try:
            worktree = repo_root(args.repo)
        except RepoError as exc:
            render_error(str(exc))
            return 2
        if has_uncommitted_changes(worktree):
            ui.render_repo_warning(worktree)
        artifacts = artifacts_dir(session)
    elif args.worktree is None:
        # Kept after the session so edits stay inspectable; never the repo root itself.
        worktree = Path.cwd() / WORKSPACES_DIRNAME / session
        worktree.mkdir(parents=True, exist_ok=True)
    else:
        worktree = Path(args.worktree).resolve()

    sandbox: DockerSandbox | None = None
    if args.docker:
        try:
            require_docker()
            sandbox = DockerSandbox(
                worktree=worktree,
                image=args.docker_image,
                session=session,
                allow_dangerous=args.allow_dangerous,
            )
            sandbox.start()
        except SandboxError as exc:
            render_error(str(exc))
            return 1
        if args.allow_dangerous:
            ui.render_danger_warning(worktree)

    # The one place the interactive CLI picks a tool-log level; Harbor stays at "off".
    display.LEVEL = "detail"
    config = _session_config(
        worktree=worktree,
        session=session,
        shell_anything=args.shell_anything,
        allow_dangerous=args.allow_dangerous,
        sandbox=sandbox,
        artifacts=artifacts,
    )
    try:
        repl(
            graph=make_graph(config),
            config=config,
            model_name=resolve_model_name(build_models(config).planner),
            commands=[ToolOutputCommand()],
        )
    finally:
        if sandbox is not None:
            sandbox.close()
        ui.render_workspace(worktree)
        if artifacts is not None:
            ui.render_artifacts(artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
