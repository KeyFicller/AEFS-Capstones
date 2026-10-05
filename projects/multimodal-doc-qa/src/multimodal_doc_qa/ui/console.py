"""Rich renderers for the interactive CLI. The only module that writes to the console."""

from pathlib import Path
from typing import Protocol

from prompt_toolkit.formatted_text import FormattedText
from repl_console import render_error as render_error
from repl_console import render_reply as render_reply
from repl_console import working as working
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from multimodal_doc_qa.schemas import page_id

CONSOLE = Console()

_TEXT_PREVIEW = 800


class SourceLine(Protocol):
    """What :func:`render_sources` needs from one citation."""

    doc_id: str
    page: int
    pdf: Path | None
    txt: Path | None
    png: Path
    text: str | None
    bbox: object


def echo(message: str) -> None:
    """One status line. Ingest and eval use this; the REPL uses the panels below."""
    CONSOLE.print(message)


def mode_models(mode: str, *, vision: str, ocr: str, describer: str) -> str:
    """The model names the status bar shows for ``mode``, and no others.

    ``summary`` is the describe arm: the VLM that wrote the page text, then the
    text embedder that retrieves it.
    """
    if mode in {"vision", "pool"}:
        return vision
    if mode == "ocr":
        return ocr
    if mode == "summary":
        return f"{describer}  {ocr}"
    raise ValueError(f"mode must be one of {', '.join(_MODES)}, got {mode!r}")


_MODES = ("vision", "ocr", "pool", "summary")


def other_mode(mode: str) -> str:
    """The retrieval path Shift-Tab switches to. One step along ``vision → ocr → pool → summary``."""
    try:
        return _MODES[(_MODES.index(mode) + 1) % len(_MODES)]
    except ValueError as exc:
        raise ValueError(f"mode must be one of {', '.join(_MODES)}, got {mode!r}") from exc


def status_bar(mode: str, embedder: str) -> FormattedText:
    """The line under the prompt: which retrieval path is active."""
    return FormattedText(
        [
            ("", " retrieval  "),
            ("bold", mode),
            ("", f"    {embedder}    shift-tab switches"),
        ]
    )


def render_sources(materials: list[SourceLine]) -> None:
    """One panel per turn: only the files that actually back each citation."""
    if not materials:
        CONSOLE.print("[yellow]sources  none[/]")
        return
    blocks: list[str] = []
    for item in materials:
        lines = [f"[bold]{escape(page_id(item.doc_id, item.page))}[/]"]
        if item.pdf is not None:
            lines.append(f"[bold]pdf[/]   {escape(str(item.pdf))}")
        if item.txt is not None:
            lines.append(f"[bold]txt[/]   {escape(str(item.txt))}")
        if item.png.is_file():
            lines.append(f"[bold]png[/]   {escape(str(item.png))}")
        if item.text:
            lines.append(f"[bold]text[/]  {escape(_clip(item.text))}")
        if item.bbox is not None:
            lines.append(f"[bold]bbox[/]  {escape(str(item.bbox))}")
        blocks.append("\n".join(lines))
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
