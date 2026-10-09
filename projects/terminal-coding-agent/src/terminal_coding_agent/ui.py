"""rich renderers for the interactive CLI. The only module that writes to the console."""

import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

# Imported for its side effect: it switches input() from the tty's canonical mode
# to a line editor. Canonical mode erases one BYTE per backspace, so a 3-byte CJK
# character takes three presses and leaves stray bytes on screen ("残留").
# macOS uses libedit, Linux GNU readline; both handle wide characters correctly.
with suppress(ImportError):
    import readline  # noqa: F401  side effect: line editor instead of canonical tty mode

# Module-level so tests can swap it for a file-backed Console.
CONSOLE = Console()

# \1..\2 tell readline these escapes occupy no columns; without them the colour
# codes would count toward the prompt width and shift the line.
APPROVAL_PROMPT = "\001\033[1;33m\002approve? [y/n] › \001\033[0m\002"
PLAIN_APPROVAL_PROMPT = "approve? [y/n] › "
QUESTION_PROMPT = "\001\033[1;36m\002answer › \001\033[0m\002"
PLAIN_QUESTION_PROMPT = "answer › "

# Tool output: the framework prints this marker, the tool supplies the rest.
TOOL_MARKER = "⚙"

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


def ask_approval(payload: Mapping[str, Any], *, show_plan: bool = True) -> str:
    """Show the plan and read a decision. Distinct prompt from `ask()` on purpose.

    `show_plan=False` is for a caller that already printed this plan — the turn's own
    Tasks panel stays as scrollback — so that only the prompt is owed. The payload is
    byte-identical to what `_announce_todos` rendered, so re-printing it just doubles it.
    """
    if show_plan:
        CONSOLE.print(
            Panel(
                escape(str(payload.get("plan", ""))),
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
        f"[bold]{index})[/] {escape(option)}" for index, option in enumerate(options, start=1)
    )
    CONSOLE.print(
        Panel(
            f"{escape(str(payload.get('question', '')))}\n\n{body}",
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
        # isdecimal(), not isdigit(): isdigit() also accepts characters `int()` rejects
        # (superscripts), which would raise ValueError out of the prompt.
        if answer.isdecimal() and 1 <= int(answer) <= len(options):
            return {"answer": options[int(answer) - 1], "cancelled": False}
        return {"answer": answer, "cancelled": False}


def render_tool_call(
    *,
    name: str,
    target: str,
    summary: str = "",
    body: str = "",
    error: str | None = None,
    header_only: bool = False,
) -> None:
    """One tool call: a single `⚙ name target (summary)` header, then the body.

    The header is always one line — the summary rides in its parentheses. `header_only`
    is the `simple` level, where the body (the detail) is left out; `detail` prints it,
    as markdown the tool chose, rendering exactly like an agent reply. An `error` result
    is printed red instead of the header's parentheses. Name, target, summary and error
    are tool text, so they are escaped, never interpolated as markup.
    """
    head = f"[bold cyan]{escape(name)}[/]" + (f"  {escape(target)}" if target else "")
    if error is not None:
        CONSOLE.print(f"[dim]{TOOL_MARKER}[/] {head}  [red]{escape(error)}[/]")
        return
    tail = f"  [dim]({escape(summary)})[/]" if summary else ""
    CONSOLE.print(f"[dim]{TOOL_MARKER}[/] {head}{tail}")
    if not header_only and body.strip():
        CONSOLE.print(Markdown(body))


def render_budget(state: Mapping[str, Any]) -> None:
    """One-line footer: how many turns ran and why it stopped.

    Token and cost totals are deliberately absent: usage accounting is unreliable
    (missing/incomplete provider metadata), so surfacing it invites wrong numbers.
    """
    CONSOLE.print(
        f"[bold]turns[/] {state.get('turns', 0)}   [bold]stop[/] {state.get('stop_reason') or 'ok'}"
    )
