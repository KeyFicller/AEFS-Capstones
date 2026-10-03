import os
from pathlib import Path

from terminal_coding_agent.config import load_local_env


def _env_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "local.env"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_local_env_reads_plain_and_exported_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TCA_PLAIN", raising=False)
    monkeypatch.delenv("TCA_EXPORTED", raising=False)
    path = _env_file(tmp_path, "TCA_PLAIN=1\nexport TCA_EXPORTED=2\n")

    load_local_env(path)

    assert os.environ["TCA_PLAIN"] == "1"
    assert os.environ["TCA_EXPORTED"] == "2"


def test_load_local_env_strips_only_a_matched_quote_pair(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TCA_QUOTED", raising=False)
    monkeypatch.delenv("TCA_LOPSIDED", raising=False)
    path = _env_file(tmp_path, 'TCA_QUOTED="secret"\nTCA_LOPSIDED=abc"\n')

    load_local_env(path)

    assert os.environ["TCA_QUOTED"] == "secret"
    assert os.environ["TCA_LOPSIDED"] == 'abc"'


def test_load_local_env_keeps_existing_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TCA_KEEP", "real")
    path = _env_file(tmp_path, "TCA_KEEP=from-file\n")

    load_local_env(path)

    assert os.environ["TCA_KEEP"] == "real"


def test_load_local_env_is_best_effort(tmp_path: Path) -> None:
    """A missing or undecodable file must not stop startup."""
    load_local_env(tmp_path / "missing.env")

    undecodable = tmp_path / "bad.env"
    undecodable.write_bytes(b"\xff\xfeKEY=value")
    load_local_env(undecodable)
