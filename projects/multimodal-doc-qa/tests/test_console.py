"""The CLI's only console writer, and the one place interpolated text meets Rich markup.

A ``[`` in an exception message, a document's text, or a path is otherwise parsed as a
style tag: it either corrupts the panel or raises ``MarkupError`` while rendering an error.
"""

from pathlib import Path
from types import SimpleNamespace

from multimodal_doc_qa.ui import console
from multimodal_doc_qa.ui.console import banner, render_error, render_sources
from rich.console import Console


def _sink(monkeypatch) -> Console:
    """A recording console in place of the module's live one: no terminal required."""
    recording = Console(record=True, width=200)
    monkeypatch.setattr(console, "CONSOLE", recording)
    return recording


def test_render_error_shows_bracketed_text_literally(monkeypatch) -> None:
    """Unescaped, ``[b]`` is a bold tag and the message silently loses its characters."""
    sink = _sink(monkeypatch)

    render_error("KeyError: 'a[b]c'")

    assert "a[b]c" in sink.export_text()


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


def test_banner_shows_bracketed_names_literally(monkeypatch) -> None:
    sink = _sink(monkeypatch)

    banner(artifacts="/tmp/[a]", mode="vision", models="m[b]")

    text = sink.export_text()
    assert "/tmp/[a]" in text and "m[b]" in text


def test_mode_models_names_only_that_arms_models() -> None:
    from multimodal_doc_qa.ui.console import mode_models

    assert mode_models("vision", vision="col", ocr="bge", describer="vlm") == "col"
    assert mode_models("pool", vision="col", ocr="bge", describer="vlm") == "col"
    assert mode_models("ocr", vision="col", ocr="bge", describer="vlm") == "bge"
    assert mode_models("summary", vision="col", ocr="bge", describer="vlm") == "vlm  bge"
