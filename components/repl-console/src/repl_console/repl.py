"""The REPL loop and the only module in this package that writes to the console."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import FormattedText

    _HAVE_PTK = True
except ImportError:  # pragma: no cover - environments without prompt_toolkit
    _HAVE_PTK = False

from repl_console.commands import Command, CommandContext, Registry
from repl_console.shorthands import (
    AttachmentError,
    build_task_message,
    completion_candidates,
    parse,
)

CONSOLE = Console()
PLAIN_PROMPT = "you › "
PROMPT = "\001\033[1;32m\002you › \001\033[0m\002"
WORKING_MESSAGE = "working…"


class EndSessionError(Exception):
    """Raised by `on_task` when the session must end, for example EOF at a nested prompt."""


def banner(*, title: str, info: Mapping[str, str]) -> None:
    body = Text.from_markup(
        "\n".join(f"[bold]{escape(key)}[/]  {escape(value)}" for key, value in info.items())
    )
    CONSOLE.print(Panel(body, title=title, border_style="cyan"))


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif part.get("type") == "image_url":
                parts.append("[image]")
        return "\n".join(parts)
    return ""


def reply_panel(content: Any) -> Panel:
    return Panel(Markdown(_content_text(content)), title="agent", border_style="green")


def render_reply(content: Any) -> None:
    CONSOLE.print(reply_panel(content))


def render_error(message: str) -> None:
    CONSOLE.print(Panel(escape(message), title="error", border_style="red"))


def render_shell(*, exit_code: int | None, output: str) -> None:
    title = "shell" if exit_code is None else f"shell (exit {exit_code})"
    body = Text(output.rstrip("\n") or "(no output)")
    CONSOLE.print(Panel(body, title=title, border_style="dim", expand=False))


def render_local(text: str) -> None:
    CONSOLE.print(Text(text))


@contextmanager
def working() -> Iterator[None]:
    """Show ``working…`` while the caller runs one turn. A no-op off a terminal."""
    if not CONSOLE.is_interactive:
        yield
        return
    spinner = Spinner("dots", text=Text(WORKING_MESSAGE, style="bold cyan"))
    with Live(
        spinner,
        console=CONSOLE,
        auto_refresh=True,
        refresh_per_second=10,
        transient=True,
        redirect_stdout=False,
        redirect_stderr=False,
    ):
        yield


def run_shell(command: str, root: Path, timeout: int) -> tuple[int | None, str]:
    """Run `command` in `root`. `None` exit code means the process group was killed."""
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        return None, f"timed out after {timeout}s"
    return process.returncode, stdout + stderr


class Repl:
    def __init__(
        self,
        *,
        title: str,
        info: Mapping[str, str],
        root: Path,
        on_task: Callable[[HumanMessage], None],
        extra_dirs: Sequence[Path] = (),
        commands: Sequence[Command] = (),
        toolbar: Callable[[], Any] | None = None,
        key_bindings: object | None = None,
        shell_timeout: int = 60,
        text_token_limit: int = 4000,
        on_ready: Callable[[], None] | None = None,
        stdin: object | None = None,
        output: object | None = None,
    ) -> None:
        self._title = title
        self._info = info
        self._root = root
        self._extra_dirs = tuple(extra_dirs)
        self._on_task = on_task
        self._registry = Registry(commands)
        self._toolbar = toolbar
        self._key_bindings = key_bindings
        self._shell_timeout = shell_timeout
        self._text_token_limit = text_token_limit
        self._on_ready = on_ready
        self._input = stdin
        self._output = output
        self._session = self._make_session()

    def run(self) -> None:
        banner(title=self._title, info=self._info)
        try:
            if self._on_ready is not None:
                self._on_ready()
        except EndSessionError:
            return
        ctx = CommandContext(root=self._root, extra_dirs=self._extra_dirs)
        while True:
            try:
                text = self._read_line().strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not text:
                continue
            try:
                message = self._dispatch(text, ctx)
            except EndSessionError:
                return
            except AttachmentError as exc:
                render_error(str(exc))
                continue
            except OSError as exc:
                render_error(str(exc))
                continue
            if message is None:
                continue
            try:
                self._on_task(message)
            except EndSessionError:
                return
            except KeyboardInterrupt:
                render_error("turn cancelled")
            except Exception as exc:  # noqa: BLE001 - one bad turn must not end the session
                render_error(str(exc))

    def _make_session(self) -> Any | None:
        if not _HAVE_PTK:
            return None
        if self._input is None and not sys.stdin.isatty():
            return None
        return PromptSession(
            completer=_Completer(
                roots=(self._root, *self._extra_dirs),
                command_names=self._registry.names(),
            ),
            bottom_toolbar=self._toolbar,
            key_bindings=self._key_bindings,
            input=self._input,
            output=self._output,
        )

    def _read_line(self) -> str:
        if self._session is not None:
            return self._session.prompt(FormattedText([("bold ansigreen", "you › ")]))
        return input(PROMPT if sys.stdin.isatty() else PLAIN_PROMPT)

    def _dispatch(self, text: str, ctx: CommandContext) -> HumanMessage | None:
        """Return the task message, or None when the line was handled locally."""
        parsed = parse(text)
        if parsed.kind == "shell":
            exit_code, output = run_shell(parsed.text, self._root, self._shell_timeout)
            render_shell(exit_code=exit_code, output=output)
            return None
        if parsed.kind == "command":
            outcome = self._registry.dispatch(parsed.command or "", parsed.text, ctx)
            if outcome is None:
                render_error(f"unknown command: /{parsed.command}")
                return None
            if outcome.message is not None:
                render_local(outcome.message)
            if outcome.quit:
                raise EndSessionError
            if outcome.prompt is None:
                return None
            return HumanMessage(content=outcome.prompt)
        return build_task_message(
            parsed,
            self._root,
            self._extra_dirs,
            token_limit=self._text_token_limit,
        )


if _HAVE_PTK:

    class _Completer(Completer):
        def __init__(self, roots: tuple[Path, ...], command_names: tuple[str, ...]) -> None:
            self._roots = roots
            self._command_names = command_names

        def get_completions(self, document: Any, complete_event: Any) -> Iterator[Any]:
            text = document.text_before_cursor
            for candidate in completion_candidates(text, self._roots, self._command_names):
                index = text.rfind(candidate[0])
                yield Completion(candidate, start_position=index - len(text))


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    process.communicate()
