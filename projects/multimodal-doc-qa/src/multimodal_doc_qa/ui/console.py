"""Rich renderers for the interactive CLI. The only module that writes to the console."""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from prompt_toolkit.formatted_text import AnyFormattedText, FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from multimodal_doc_qa.schemas import page_id

CONSOLE = Console()

PLAIN_PROMPT = "you › "

_TEXT_PREVIEW = 800


class SourceLine(Protocol):
    """What :func:`render_sources` needs from one citation."""

    doc_id: str
    page: int
    pdf: Path | None
    png: Path
    text: str | None
    bbox: object


def echo(message: str) -> None:
    """One status line. Ingest and eval use this; the REPL uses the panels below."""
    CONSOLE.print(message)


def banner(*, artifacts: str, mode: str, embedder: str, answerer: str) -> None:
    """Opening panel: which artifacts, which arm, which models."""
    body = Text.from_markup(
        f"[bold]artifacts[/] {artifacts}\n"
        f"[bold]mode[/]      {mode}\n"
        f"[bold]embedder[/]  {embedder}\n"
        f"[bold]answerer[/]  {answerer}"
    )
    CONSOLE.print(Panel(body, title="multimodal-doc-qa", border_style="cyan"))


def other_mode(mode: str) -> str:
    """The retrieval path Shift-Tab switches to."""
    if mode == "vision":
        return "ocr"
    if mode == "ocr":
        return "vision"
    raise ValueError(f"mode must be vision or ocr, got {mode!r}")


def status_bar(mode: str, embedder: str) -> FormattedText:
    """The line under the prompt: which retrieval path is active."""
    return FormattedText([
        ("", " retrieval  "),
        ("bold", mode),
        ("", f"    {embedder}    shift-tab switches"),
    ])


class Prompt:
    """One line of input. On a terminal, Shift-Tab runs ``on_switch`` and the status bar redraws.

    Piped stdin uses ``input()`` and has no status bar: there is no key to bind.
    """

    def __init__(
        self,
        on_switch: Callable[[], None],
        status: Callable[[], AnyFormattedText],
        *,
        input: object | None = None,
        output: object | None = None,
    ) -> None:
        self._piped = input is None and not sys.stdin.isatty()
        if self._piped:
            return
        bindings = KeyBindings()

        @bindings.add("s-tab")
        def _switch(event) -> None:
            on_switch()
            event.app.invalidate()

        self._session: PromptSession[str] = PromptSession(
            key_bindings=bindings,
            bottom_toolbar=status,
            input=input,
            output=output,
        )

    def ask(self) -> str:
        """One user line. Raises EOFError on Ctrl-D, KeyboardInterrupt on Ctrl-C."""
        if self._piped:
            return input(PLAIN_PROMPT)
        return self._session.prompt(FormattedText([("bold ansigreen", "you › ")]))


def render_reply(text: str) -> None:
    CONSOLE.print(Panel(Markdown(text), title="agent", border_style="green"))


def render_error(message: str) -> None:
    CONSOLE.print(Panel(message, title="error", border_style="red"))


def render_sources(materials: list[SourceLine]) -> None:
    """One panel per turn: the pdf, page image, and OCR text behind each citation."""
    if not materials:
        CONSOLE.print("[yellow]sources  none[/]")
        return
    blocks: list[str] = []
    for item in materials:
        pdf = str(item.pdf) if item.pdf is not None else f"{item.doc_id}.pdf  [yellow]not in artifacts[/]"
        png = str(item.png) if item.png.is_file() else f"{item.png}  [yellow]missing[/]"
        text = _clip(item.text) if item.text else "[yellow]no OCR text for this page[/]"
        blocks.append(
            f"[bold]{page_id(item.doc_id, item.page)}[/]\n"
            f"[bold]pdf[/]   {pdf}\n"
            f"[bold]png[/]   {png}\n"
            f"[bold]text[/]  {text}\n"
            f"[bold]bbox[/]  {item.bbox}"
        )
    CONSOLE.print(Panel("\n\n".join(blocks), title="sources", border_style="cyan", expand=False))


def render_budget(
    *,
    rounds: int,
    max_rounds: int,
    calls: int,
    max_calls: int,
    tokens: int,
    max_tokens: int,
    elapsed: float,
    max_seconds: float,
    stop_reason: str,
) -> None:
    """One-line footer for the turn."""
    CONSOLE.print(
        f"[bold]rounds[/] {rounds}/{max_rounds}   "
        f"[bold]calls[/] {calls}/{max_calls}   "
        f"[bold]tokens[/] {tokens}/{max_tokens}   "
        f"[bold]stop[/] {stop_reason or 'ok'}   "
        f"{elapsed:.1f}s/{max_seconds:.0f}s"
    )


def _clip(text: str) -> str:
    if len(text) <= _TEXT_PREVIEW:
        return text
    return f"{text[:_TEXT_PREVIEW]}  ... ({len(text)} chars)"
