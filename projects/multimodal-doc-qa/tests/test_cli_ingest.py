"""``ingest`` and ``ask`` have to agree about what is on disk, and ``ingest`` has to
actually put something there.

The failure this file exists to catch is a silent one: an ``ingest`` that renders pages,
builds no vectors, and saves the empty index exits 0 and prints a success line, and the
only symptom is that every later ``ask`` reports an empty page pool. Asserting that the
commands are *registered* cannot see that, so the assertions here load the artifacts back
and check their contents.

Weights and the answerer are stubbed -- these tests are about the file contract, not about
what a checkpoint scores. The encoder stub is deliberately trivial: it makes every page score
equally, which is enough to prove the query reached an index that has pages in it.
"""

import os
from pathlib import Path

import pymupdf
import pytest
import torch
from typer.testing import CliRunner

from multimodal_doc_qa.cli import app, cited_materials
from multimodal_doc_qa.config import Settings, artifact_paths, load_local_env
from multimodal_doc_qa.schemas import Citation
from multimodal_doc_qa.index.maxsim import MultiVectorIndex

DOC_TEXT = "EMEA margin was 16.8%"


class _StubVisionEncoder:
    """Always returns the same single patch vector, so every page scores identically."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.model = self

    def encode_images(self, images: list[object]) -> list[torch.Tensor]:
        return [torch.tensor([[1.0, 0.0]]) for _ in images]

    def encode_query(self, text: str) -> torch.Tensor:
        return torch.tensor([[1.0, 0.0]])


class _StubOcrEmbedder:
    calls: list[tuple[object, ...]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        type(self).calls.append(args)

    def encode(self, texts: list[str]) -> torch.Tensor:
        return torch.ones((len(texts), 2))


class _StubStructured:
    """Returns an empty instance of whatever schema it was bound to.

    Empty means "no follow-ups" and "nothing unsupported", so the run takes the shortest
    honest path: plan falls back to the question, assess is satisfied, verify passes.
    """

    def __init__(self, schema: object) -> None:
        self.schema = schema

    def invoke(self, messages: list[dict]) -> object:
        return self.schema()  # type: ignore[operator]


class _StubChat:
    def with_structured_output(self, schema: object) -> _StubStructured:
        return _StubStructured(schema)


class _StubSynth:
    """Answers from the pool's first page, so the citation is a real retrieved page."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def synthesize(self, messages: object, page_ids: list[str], render_dir: Path) -> object:
        from multimodal_doc_qa.schemas import Answer, Citation

        doc_id, _, page = page_ids[0].partition("/p")
        return Answer(
            text=DOC_TEXT,
            citations=[Citation(doc_id=doc_id, page=int(page), bbox=None)],
        )


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every command at a throwaway artifacts root and stub the model weights."""
    root = tmp_path / "artifacts"
    monkeypatch.setenv("MDQ_ARTIFACTS_DIR", str(root))
    monkeypatch.setattr(
        "multimodal_doc_qa.embed.encoder.MultiVectorEncoder", _StubVisionEncoder
    )
    monkeypatch.setattr("multimodal_doc_qa.retrievers.text.OcrEmbedder", _StubOcrEmbedder)
    _StubOcrEmbedder.calls = []
    return root


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """One PDF with a text layer, so the OCR path reads the layer instead of Tesseract."""
    directory = tmp_path / "corpus"
    directory.mkdir()
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), DOC_TEXT)
        pdf.save(directory / "doc000.pdf")
    return directory


def _invoke(*args: str, stdin: str | None = None) -> object:
    return CliRunner().invoke(app, list(args), input=stdin)


# ------------------------------------------------------------------ wiring


def test_load_local_env_sets_missing_keys_and_keeps_existing_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "local.env"
    env_file.write_text('DEEPSEEK_API_KEY="from-file"\nALREADY=from-file\n# comment\n\n')
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("ALREADY", "from-shell")

    load_local_env(env_file)

    assert os.environ["DEEPSEEK_API_KEY"] == "from-file"
    assert os.environ["ALREADY"] == "from-shell"


def test_the_help_lists_both_commands() -> None:
    result = _invoke("--help")

    assert result.exit_code == 0
    assert "ingest" in result.stdout and "ask" in result.stdout and "chat" in result.stdout


# ------------------------------------------------------------------ ingest


def test_ingest_saves_a_vision_index_that_holds_the_pages(corpus: Path, artifacts: Path) -> None:
    """The regression that matters: an index with no vectors looks exactly like success."""
    result = _invoke("ingest", "--corpus", str(corpus))

    assert result.exit_code == 0, result.stdout
    _, vision_path, _ = artifact_paths(artifacts)
    index = MultiVectorIndex.load(vision_path)
    assert index.search(torch.tensor([[1.0, 0.0]]), k=5), "the saved index has no pages"


def test_ingest_renders_the_pages_it_indexed(corpus: Path, artifacts: Path) -> None:
    _invoke("ingest", "--corpus", str(corpus))

    render_dir, _, _ = artifact_paths(artifacts)
    assert (render_dir / "doc000" / "p000.png").is_file()


def test_ingest_saves_an_ocr_index_with_chunk_text(corpus: Path, artifacts: Path) -> None:
    """The OCR arm has to be usable without re-ingesting, or the ablation is unrunnable."""
    _invoke("ingest", "--corpus", str(corpus))

    _, _, ocr_path = artifact_paths(artifacts)
    saved = torch.load(ocr_path, weights_only=True)
    assert list(saved) == ["doc000"]
    assert any(DOC_TEXT in text for text, _ in saved["doc000"]["chunks"])


def test_ingest_refuses_an_empty_corpus(tmp_path: Path, artifacts: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    result = _invoke("ingest", "--corpus", str(empty))

    assert result.exit_code == 1


def test_ingest_writes_where_ask_reads(corpus: Path, artifacts: Path) -> None:
    """``--out`` and the default must not be two different layouts."""
    other = artifacts.parent / "elsewhere"

    _invoke("ingest", "--corpus", str(corpus), "--out", str(other))

    render_dir, vision_path, ocr_path = artifact_paths(other)
    assert vision_path.is_file() and ocr_path.is_file()
    assert (render_dir / "doc000" / "p000.png").is_file()


# ------------------------------------------------------------------ ask


def test_ask_rejects_an_unknown_mode_without_loading_anything(artifacts: Path) -> None:
    result = _invoke("ask", "anything", "--mode", "telepathy")

    assert result.exit_code != 0


def test_ask_answers_from_the_pages_ingest_indexed(
    corpus: Path, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end-to-end contract: ingest, then ask, with no re-encoding in between."""
    monkeypatch.setattr("multimodal_doc_qa.synth.answer.build_chat_model", lambda *a, **k: _StubChat())
    monkeypatch.setattr("multimodal_doc_qa.synth.answer.AnswerSynthesizer", _StubSynth)
    _invoke("ingest", "--corpus", str(corpus))

    result = _invoke("ask", "what was the EMEA margin?")

    assert result.exit_code == 0, result.stdout
    assert DOC_TEXT in result.stdout
    assert "doc000/p000" in result.stdout


def test_ask_reports_no_answer_when_nothing_was_ingested(
    artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ask against a missing index must say so, not raise from inside torch.load."""
    monkeypatch.setattr("multimodal_doc_qa.synth.answer.build_chat_model", lambda *a, **k: _StubChat())
    monkeypatch.setattr("multimodal_doc_qa.synth.answer.AnswerSynthesizer", _StubSynth)

    result = _invoke("ask", "what was the EMEA margin?")

    assert result.exit_code == 1
    assert "ingest" in result.stdout


def test_ingest_and_ocr_ask_load_the_configured_ocr_embedder(
    corpus: Path, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both ends of the OCR arm must embed with the same checkpoint and device.

    A query vector from a different model than the one that wrote the index scores silently wrong.
    """
    monkeypatch.setenv("MDQ_OCR_EMBEDDER_MODEL", "org/fake-text")
    monkeypatch.setattr("multimodal_doc_qa.synth.answer.build_chat_model", lambda *a, **k: _StubChat())
    monkeypatch.setattr("multimodal_doc_qa.synth.answer.AnswerSynthesizer", _StubSynth)
    device = Settings().device

    _invoke("ingest", "--corpus", str(corpus))
    result = _invoke("ask", "what was the EMEA margin?", "--mode", "ocr")

    assert result.exit_code == 0, result.stdout
    assert _StubOcrEmbedder.calls == [("org/fake-text", device), ("org/fake-text", device)]


def test_the_ocr_arm_retrieves_text_chunks(corpus: Path, artifacts: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multimodal_doc_qa.synth.answer.build_chat_model", lambda *a, **k: _StubChat())
    monkeypatch.setattr("multimodal_doc_qa.synth.answer.AnswerSynthesizer", _StubSynth)
    _invoke("ingest", "--corpus", str(corpus))

    result = _invoke("ask", "what was the EMEA margin?", "--mode", "ocr")

    assert result.exit_code == 0, result.stdout
    assert "doc000/p000" in result.stdout


def test_cited_materials_resolve_pdf_png_and_page_text(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    png = artifacts / "render" / "Test" / "p024.png"
    png.parent.mkdir(parents=True)
    png.write_bytes(b"png")
    pdf = artifacts / "Test.pdf"
    pdf.write_bytes(b"%PDF")

    found = cited_materials(
        [Citation(doc_id="Test", page=24)],
        artifacts,
        {("Test", 24): "Flamingo objectives"},
    )

    assert found[0].pdf == pdf
    assert found[0].png == png
    assert found[0].text == "Flamingo objectives"


def test_cited_materials_leave_pdf_and_text_empty_when_absent(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    found = cited_materials([Citation(doc_id="Test", page=1)], artifacts, {})

    assert found[0].pdf is None
    assert found[0].text is None
    assert found[0].png == artifacts / "render" / "Test" / "p001.png"


def test_repl_prints_the_answer_and_the_cited_files(
    corpus: Path, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two turns, one process: the indexes stay loaded and each turn names its files."""
    monkeypatch.setattr("multimodal_doc_qa.synth.answer.build_chat_model", lambda *a, **k: _StubChat())
    monkeypatch.setattr("multimodal_doc_qa.synth.answer.AnswerSynthesizer", _StubSynth)
    _invoke("ingest", "--corpus", str(corpus))

    result = _invoke("chat", stdin="what was the EMEA margin?\nand the other page?\n:q\n")

    assert result.exit_code == 0, result.stdout
    assert result.stdout.count(DOC_TEXT) >= 2
    assert "doc000/p000" in result.stdout
    assert "png" in result.stdout and "text" in result.stdout


def test_no_command_opens_the_repl(artifacts: Path) -> None:
    """Startup with no subcommand reads artifacts and refuses a missing index."""
    result = _invoke(stdin=":q\n")

    assert result.exit_code == 1
    assert "ingest" in result.stdout


def test_no_command_can_select_the_ocr_path(artifacts: Path) -> None:
    """The bare CLI takes the same two retrieval paths as ask and chat."""
    result = _invoke("--mode", "ocr", stdin=":q\n")

    assert result.exit_code == 1
    assert "OCR" in result.stdout


def test_shift_tab_switches_the_retrieval_path_on_the_status_bar() -> None:
    """Shift-Tab flips the path the status bar is showing, before the line is submitted."""
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from multimodal_doc_qa.ui.console import Prompt, other_mode, status_bar

    mode = {"value": "vision"}

    def switch() -> None:
        mode["value"] = other_mode(mode["value"])

    with create_pipe_input() as pipe:
        prompt = Prompt(
            on_switch=switch,
            status=lambda: status_bar(mode["value"], "BAAI/bge-small-en-v1.5"),
            input=pipe,
            output=DummyOutput(),
        )
        pipe.send_text("\x1b[Zhi\r")
        assert prompt.ask() == "hi"

    assert mode["value"] == "ocr"
    shown = "".join(piece for _, piece in status_bar(mode["value"], "BAAI/bge-small-en-v1.5"))
    assert "ocr" in shown and "shift-tab" in shown


def test_the_two_arms_point_at_the_same_render_dir(artifacts: Path) -> None:
    """Both arms show pages to the model, so both need the same rendered PNGs."""
    settings = Settings()

    assert artifact_paths(settings.artifacts_dir)[0] == settings.render_dir
