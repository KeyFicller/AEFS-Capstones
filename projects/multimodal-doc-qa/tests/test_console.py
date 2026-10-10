"""The CLI's only console writer, and the one place interpolated text meets Rich markup.

A ``[`` in an exception message, a document's text, or a path is otherwise parsed as a
style tag: it either corrupts the panel or raises ``MarkupError`` while rendering an error.
"""

from pathlib import Path
from types import SimpleNamespace

from multimodal_doc_qa.ui import console
from multimodal_doc_qa.ui.console import render_error, render_sources
from repl_console import repl
from rich.console import Console


def _sink(monkeypatch) -> Console:
    """A recording console in place of the module's live one: no terminal required."""
    recording = Console(record=True, width=200)
    monkeypatch.setattr(console, "CONSOLE", recording)
    return recording


def test_render_error_shows_bracketed_text_literally(monkeypatch) -> None:
    """Unescaped, ``[b]`` is a bold tag and the message silently loses its characters."""
    recording = Console(record=True, width=200)
    monkeypatch.setattr(repl, "CONSOLE", recording)

    render_error("KeyError: 'a[b]c'")

    assert "a[b]c" in recording.export_text()


def test_render_sources_escapes_paths_and_text(monkeypatch, tmp_path: Path) -> None:
    sink = _sink(monkeypatch)
    png = tmp_path / "[missing].png"
    png.write_bytes(b"x")
    material = SimpleNamespace(
        doc_id="doc[b]",
        page=0,
        pdf=None,
        txt=None,
        png=png,
        text="row [b] value",
        bbox=None,
    )

    render_sources([material])

    text = sink.export_text()
    assert "doc[b]/p000" in text
    assert "row [b] value" in text
    assert "[missing].png" in text


def test_mode_models_names_only_that_arms_models() -> None:
    from multimodal_doc_qa.ui.console import mode_models

    assert mode_models("maxsim", embedder="col", ocr="bge", describer="vlm") == "col"
    assert mode_models("pool", embedder="col", ocr="bge", describer="vlm") == "col"
    assert mode_models("ocr", embedder="col", ocr="bge", describer="vlm") == "bge"
    assert mode_models("abstract", embedder="col", ocr="bge", describer="vlm") == "vlm  bge"
    assert mode_models("lexical", embedder="col", ocr="bge", describer="vlm") == "bm25"
    assert (
        mode_models("lexical-kw", embedder="col", ocr="bge", describer="vlm", extractor="vlm")
        == "vlm  bm25"
    )
    assert mode_models("hybrid-ocr", embedder="col", ocr="bge", describer="vlm") == "bge"
    assert mode_models("hybrid-maxsim", embedder="col", ocr="bge", describer="vlm") == "col"


def test_other_mode_cycles_through_every_arm() -> None:
    from multimodal_doc_qa.retrievers.assembly import MODES
    from multimodal_doc_qa.ui.console import other_mode

    seen = [MODES[0]]
    for _ in range(len(MODES) - 1):
        seen.append(other_mode(seen[-1]))

    assert seen == list(MODES)
    assert other_mode(MODES[-1]) == MODES[0]
