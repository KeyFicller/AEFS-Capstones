from pathlib import Path

import pytest
import typer
from multimodal_doc_qa import cli


def _command(tmp_path: Path, mode: str = "maxsim"):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    loaded = {"maxsim": "old"}
    missing: dict[str, str] = {}
    page_texts = {("old", 0): "old"}
    history = ["turn"]
    active = {"mode": mode}
    command = cli.IndexCommand(
        settings=object(),
        artifacts=artifacts,
        loaded=loaded,
        missing=missing,
        page_texts=page_texts,
        history=history,
        active=active,
    )
    return command, artifacts, loaded, missing, page_texts, history, active


def test_dot_indexes_the_artifacts_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command, artifacts, *_ = _command(tmp_path)
    seen: list[Path] = []

    def ingest(*, corpus: Path, out: Path | None) -> None:
        seen.append(corpus)

    monkeypatch.setattr(cli, "ingest", ingest)
    monkeypatch.setattr(cli, "_preload_retrievers", lambda settings: ({"maxsim": "new"}, {}))
    monkeypatch.setattr(cli, "_ocr_page_texts", lambda settings: {})

    outcome = command.run(None, ".")

    assert seen == [artifacts.resolve()]
    assert outcome.message == "using the new index"


def test_relative_path_is_under_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command, artifacts, *_ = _command(tmp_path)
    seen: list[Path] = []
    monkeypatch.setattr(cli, "ingest", lambda *, corpus, out: seen.append(corpus))
    monkeypatch.setattr(cli, "_preload_retrievers", lambda settings: ({"maxsim": "new"}, {}))
    monkeypatch.setattr(cli, "_ocr_page_texts", lambda settings: {})

    command.run(None, "papers")

    assert seen == [(artifacts / "papers").resolve()]


def test_a_path_outside_artifacts_does_not_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command, artifacts, loaded, _, page_texts, history, _ = _command(tmp_path)
    called = {"n": 0}
    monkeypatch.setattr(cli, "ingest", lambda *, corpus, out: called.__setitem__("n", 1))

    outcome = command.run(None, "..")

    assert called["n"] == 0
    assert outcome.message == "path escapes artifacts"
    assert loaded == {"maxsim": "old"}
    assert history == ["turn"]
    assert page_texts == {("old", 0): "old"}


def test_missing_argument_does_not_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command, *_ = _command(tmp_path)
    called = {"n": 0}
    monkeypatch.setattr(cli, "ingest", lambda *, corpus, out: called.__setitem__("n", 1))

    outcome = command.run(None, "  ")

    assert called["n"] == 0
    assert outcome.message == "usage: /index <corpus-dir>"


def test_ingest_exit_keeps_the_old_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command, _, loaded, _, page_texts, history, active = _command(tmp_path)

    def ingest(*, corpus: Path, out: Path | None) -> None:
        raise typer.Exit(code=1)

    monkeypatch.setattr(cli, "ingest", ingest)

    outcome = command.run(None, ".")

    assert outcome.message is None
    assert loaded == {"maxsim": "old"}
    assert history == ["turn"]
    assert page_texts == {("old", 0): "old"}
    assert active["mode"] == "maxsim"


def test_success_swaps_the_retriever_and_clears_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command, _, loaded, missing, page_texts, history, active = _command(tmp_path)
    monkeypatch.setattr(cli, "ingest", lambda *, corpus, out: None)
    monkeypatch.setattr(
        cli, "_preload_retrievers", lambda settings: ({"maxsim": "new", "ocr": "ocr"}, {"abstract": "no"})
    )
    monkeypatch.setattr(cli, "_ocr_page_texts", lambda settings: {("new", 1): "page"})

    command.run(None, ".")

    assert loaded == {"maxsim": "new", "ocr": "ocr"}
    assert missing == {"abstract": "no"}
    assert page_texts == {("new", 1): "page"}
    assert history == []
    assert active["mode"] == "maxsim"


def test_a_missing_current_mode_switches_to_the_first_loaded_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command, _, loaded, _, _, history, active = _command(tmp_path, mode="abstract")
    monkeypatch.setattr(cli, "ingest", lambda *, corpus, out: None)
    monkeypatch.setattr(
        cli, "_preload_retrievers", lambda settings: ({"ocr": "ocr"}, {"abstract": "no abstract index"})
    )
    monkeypatch.setattr(cli, "_ocr_page_texts", lambda settings: {})

    outcome = command.run(None, ".")

    assert active["mode"] == "ocr"
    assert loaded == {"ocr": "ocr"}
    assert history == []
    assert outcome.message == "using the new index (ocr)"


def test_nothing_loaded_keeps_the_old_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command, _, loaded, _, _, history, active = _command(tmp_path, mode="maxsim")
    errors: list[str] = []
    monkeypatch.setattr(cli, "ingest", lambda *, corpus, out: None)
    monkeypatch.setattr(cli, "_preload_retrievers", lambda settings: ({}, {"maxsim": "no vision index"}))
    monkeypatch.setattr(cli.ui, "render_error", errors.append)

    outcome = command.run(None, ".")

    assert errors == ["no vision index"]
    assert outcome.message is None
    assert loaded == {"maxsim": "old"}
    assert history == ["turn"]
    assert active["mode"] == "maxsim"
