"""rich renderers for the interactive CLI. The only module that writes to the console."""

import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

# Imported for its side effect: it switches input() from the tty's canonical mode
# to a line editor. Canonical mode erases one BYTE per backspace, so a 3-byte CJK
# character takes three presses and leaves stray bytes on screen ("残留").
# macOS uses libedit, Linux GNU readline; both handle wide characters correctly.
try:
    import readline  # noqa: F401
except ImportError:  # pragma: no cover - platforms without readline (e.g. Windows)
    pass

# Module-level so tests can swap it for a file-backed Console.
CONSOLE = Console()

# \1..\2 tell readline these escapes occupy no columns; without them the colour
# codes would count toward the prompt width and shift the line.
PROMPT = "\001\033[1;32m\002you › \001\033[0m\002"
PLAIN_PROMPT = "you › "
APPROVAL_PROMPT = "\001\033[1;33m\002approve? [y/n] › \001\033[0m\002"
PLAIN_APPROVAL_PROMPT = "approve? [y/n] › "
QUESTION_PROMPT = "\001\033[1;36m\002answer › \001\033[0m\002"
PLAIN_QUESTION_PROMPT = "answer › "

# Tool log: which argument identifies the call, and what its result counts as.
TOOL_MARKER = "⚙"
MAX_TARGET_CHARS = 60
_TOOL_TARGET_ARG = {
    "read_file": "path",
    "edit_file": "path",
    "tree_sitter_symbols": "path",
    "ripgrep": "pattern",
    "run_shell": "command",
    "git": "git_args",
}
_TOOL_RESULT_UNIT = {
    "read_file": "lines",
    "ripgrep": "hits",
    "tree_sitter_symbols": "symbols",
}
_EXIT_CODE_TOOLS = frozenset({"run_shell", "git"})

WORKING_MESSAGE = "working…"
TODO_TITLE = "Tasks"


class TodoPanel:
    """`todo_renderer` that redraws one panel in place while a turn runs.

    The graph calls the renderer from several nodes, so a plain `print` leaves a
    trail of panels — one per status change. Wrapping a turn in :meth:`region`
    turns those calls into in-place redraws on a terminal, and shows an animated
    "working…" spinner for the model calls in between. Off a terminal (Harbor
    logs, pipes) there is nothing to overwrite, so every state is printed and
    the log keeps the same fidelity it had before.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console if console is not None else CONSOLE
        self._live: Live | None = None
        self._panel: Panel | None = None
        self._working = False
        self._spinner = Spinner("dots", text=Text(WORKING_MESSAGE, style="bold cyan"))

    def __call__(self, text: str, version: int = 0) -> None:
        self._panel = Panel(
            text,
            title=TODO_TITLE,
            # Shown even at v0: hiding it made the count effectively invisible, since it
            # resets to 0 every turn and only a mid-turn replan bumps it.
            subtitle=f"replan v{version}",
            border_style="blue",
            expand=False,
        )
        if self._live is None:
            self._console.print(self._panel)
        else:
            self._live.refresh()

    def _renderable(self) -> RenderableType:
        body: list[RenderableType] = []
        if self._panel is not None:
            body.append(self._panel)
        if self._working:
            body.append(self._spinner)
        return Group(*body)

    @contextmanager
    def region(self) -> Iterator[None]:
        """Redraw in place for the duration of one turn; a no-op off a terminal."""
        if not self._console.is_interactive:
            yield
            return
        # Panel and spinner share ONE Live on purpose. A nested `Console.status`
        # looks tempting but breaks twice over: rich returns early for a nested
        # Live, so it never starts a refresh thread (spinner frozen), and its
        # transient cleanup moves the cursor as if it had drawn alone, smearing
        # the panel. One Live also gives the spinner its animation for free.
        with Live(
            console=self._console,
            get_renderable=self._renderable,
            auto_refresh=True,
            refresh_per_second=10,
            # Live's defaults would swap the process stdout/stderr for proxies,
            # which would capture unrelated logging and pytest output.
            redirect_stdout=False,
            redirect_stderr=False,
        ) as live:
            self._live = live
            self._panel = None  # this turn starts clean; the old panel stays as scrollback
            self._working = True
            try:
                yield
            finally:
                # Drop the spinner before Live.stop() renders for the last time,
                # so the finished turn leaves the todo panel and no "working…".
                self._working = False
                self._live = None


def _tool_target(name: str, args: Mapping[str, Any]) -> str:
    """The one argument worth showing: the file, the pattern, the command."""
    value = args.get(_TOOL_TARGET_ARG.get(name, ""))
    if isinstance(value, (list, tuple)):
        value = " ".join(str(part) for part in value)
    if value is None:
        return ""
    text = str(value).replace("\n", " ⏎ ")
    return text if len(text) <= MAX_TARGET_CHARS else text[: MAX_TARGET_CHARS - 1] + "…"


def _tool_outcome(name: str, result: str) -> str:
    """Compact descriptor of what a tool produced: exit code, hit count, diff size."""
    text = result.strip()
    if not text:
        return "no hits" if name == "ripgrep" else "no output"
    if text.startswith("Error:"):
        return text.splitlines()[0]
    if name == "edit_file":
        lines = text.splitlines()
        added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
        return f"+{added} -{removed}"
    if name in _EXIT_CODE_TOOLS:
        match = re.match(r"exit_code:\s*(-?\d+)", text)
        return f"exit {match.group(1)}" if match else "done"
    count = len(text.splitlines())
    unit = _TOOL_RESULT_UNIT.get(name, "lines")
    if count == 1 and unit.endswith("s"):
        unit = unit[:-1]
    return f"{count} {unit}"


class ToolLog:
    """`tool_renderer`: one line per executed tool call, so a turn is not a black box.

    Prints while the Tasks panel is live, so the lines scroll above it and the panel
    stays pinned at the bottom. `edit_file` already returns a unified diff, which is
    colourised here rather than inside the tool — tools stay free of presentation.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console if console is not None else CONSOLE

    def __call__(self, name: str, args: Mapping[str, Any], result: str) -> None:
        target = _tool_target(name, args)
        head = f"[cyan]{name}[/]" + (f"  {target}" if target else "")
        self._console.print(
            f"[dim]{TOOL_MARKER}[/] {head}  [dim]({_tool_outcome(name, result)})[/]"
        )
        if name == "edit_file" and result.strip() and not result.lstrip().startswith("Error:"):
            # background_color="default" keeps each line at its natural width. The
            # default theme background pads lines to 80 columns, which wraps on a
            # narrower terminal and desyncs the live panel's height accounting.
            self._console.print(
                Syntax(result, "diff", background_color="default", word_wrap=True)
            )


def banner(*, session: str, worktree: str, model: str) -> None:
    """Opening panel: which thread, which worktree, which model."""
    body = Text.from_markup(
        f"[bold]session[/]  {session}\n[bold]worktree[/] {worktree}\n[bold]model[/]    {model}"
    )
    CONSOLE.print(Panel(body, title="terminal-coding-agent", border_style="cyan"))


def ask() -> str:
    """One user line. Raises EOFError on Ctrl-D, KeyboardInterrupt on Ctrl-C.

    The prompt is handed to `input()` rather than `Console.input()`, because readline
    must own it: given the prompt, it knows the line's starting column and refuses to
    backspace past it. Printed separately (as rich does), readline treats column 0 as
    the start of the input, so over-backspacing eats the prompt itself. `\\1`/`\\2`
    mark the ANSI escapes as zero-width. Piped stdin gets the plain text instead.
    """
    return input(PROMPT if sys.stdin.isatty() else PLAIN_PROMPT)


def ask_approval(payload: Mapping[str, Any], *, show_plan: bool = True) -> str:
    """Show the plan and read a decision. Distinct prompt from `ask()` on purpose.

    `show_plan=False` is for a caller that already printed this plan — the turn's own
    Tasks panel stays as scrollback — so that only the prompt is owed. The payload is
    byte-identical to what `_announce_todos` rendered, so re-printing it just doubles it.
    """
    if show_plan:
        CONSOLE.print(
            Panel(
                str(payload.get("plan", "")),
                title="plan for approval",
                border_style="yellow",
                expand=False,
            )
        )
    prompt = APPROVAL_PROMPT if sys.stdin.isatty() else PLAIN_APPROVAL_PROMPT
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"y", "yes", "approve"}:
            return "approve"
        if answer in {"n", "no", "reject"}:
            return "reject"
        CONSOLE.print("[yellow]answer y or n[/]")


def ask_question(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Show a question with numbered options and read a choice or free text.

    A number picks that option; anything else is taken as the user's own answer.
    Raises EOFError/KeyboardInterrupt — the caller decides what a dismissal means,
    exactly as `ask_approval` leaves the plan gate's reject policy to the CLI.
    """
    options = [str(option) for option in payload.get("options") or []]
    body = "\n".join(
        f"[bold]{index})[/] {option}" for index, option in enumerate(options, start=1)
    )
    CONSOLE.print(
        Panel(
            f"{payload.get('question', '')}\n\n{body}",
            title="agent asks",
            border_style="cyan",
            expand=False,
        )
    )
    prompt = QUESTION_PROMPT if sys.stdin.isatty() else PLAIN_QUESTION_PROMPT
    while True:
        answer = input(prompt).strip()
        if not answer:
            CONSOLE.print("[cyan]type a number or your own answer[/]")
            continue
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return {"answer": options[int(answer) - 1], "cancelled": False}
        return {"answer": answer, "cancelled": False}


def render_reply(text: str) -> None:
    CONSOLE.print(Panel(Markdown(text), title="agent", border_style="green"))


def render_error(message: str) -> None:
    CONSOLE.print(Panel(message, title="error", border_style="red"))


def render_budget(state: Mapping[str, Any]) -> None:
    """One-line footer: how many turns ran and why it stopped.

    Token and cost totals are deliberately absent: usage accounting is unreliable
    (missing/incomplete provider metadata), so surfacing it invites wrong numbers.
    """
    CONSOLE.print(
        f"[bold]turns[/] {state.get('turns', 0)}   "
        f"[bold]stop[/] {state.get('stop_reason') or 'ok'}"
    )
