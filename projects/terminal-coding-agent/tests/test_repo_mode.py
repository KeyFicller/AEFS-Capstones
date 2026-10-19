"""`--repo`: work inside an existing repository, keep run state out of it."""

import io
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from terminal_coding_agent import cli, ui


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.email", "t@t"),
        ("config", "user.name", "t"),
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    return repo


class _FakeGraph:
    def update_state(self, config, values) -> None:  # noqa: ARG002
        pass

    def get_state(self, config):  # noqa: ARG002
        return SimpleNamespace(tasks=())

    def invoke(self, payload, config):  # noqa: ARG002
        return {"messages": [], "turns": 1, "tokens": 10, "cost_rmb": 0.0, "stop_reason": None}


def _capture(monkeypatch) -> io.StringIO:
    stream = io.StringIO()
    monkeypatch.setattr(
        ui, "CONSOLE", Console(file=stream, width=100, no_color=True, force_terminal=False)
    )
    return stream


def _stub(monkeypatch, called: dict) -> None:
    monkeypatch.setattr(cli, "load_local_env", lambda path: None)
    monkeypatch.setattr(cli, "make_graph", lambda config: _FakeGraph())
    monkeypatch.setattr(cli, "build_models", lambda config: SimpleNamespace(planner="stub"))
    monkeypatch.setattr(cli, "resolve_model_name", lambda model: "stub-model")
    monkeypatch.setattr(cli, "repl", lambda **kwargs: called.update(kwargs))


def test_repo_root_resolves_a_subdirectory(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    nested = repo / "pkg"
    nested.mkdir()

    assert cli.repo_root(str(nested)) == repo.resolve()


def test_repo_root_rejects_a_plain_directory(tmp_path: Path) -> None:
    with pytest.raises(cli.RepoError):
        cli.repo_root(str(tmp_path))


def test_repo_and_worktree_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--repo", str(tmp_path), "--worktree", str(tmp_path)])


def test_main_rejects_a_directory_that_is_not_a_git_repo(tmp_path: Path, monkeypatch) -> None:
    _capture(monkeypatch)
    _stub(monkeypatch, {})

    assert cli.main(["--repo", str(tmp_path)]) == 2


def test_main_repo_edits_in_place_and_keeps_state_outside(tmp_path: Path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _capture(monkeypatch)
    called: dict = {}
    _stub(monkeypatch, called)

    assert cli.main(["--repo", str(repo), "--session", "s1"]) == 0

    configurable = called["config"]["configurable"]
    assert configurable["worktree"] == str(repo.resolve())
    artifacts = Path(configurable["artifacts_dir"])
    assert artifacts == tmp_path / "cache" / cli.CACHE_DIRNAME / "s1"
    assert artifacts.is_dir()
    assert repo not in artifacts.parents, "run state must not live inside the repository"


def test_main_repo_warns_about_uncommitted_changes(tmp_path: Path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    (repo / "app.py").write_text("x = 2\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    stream = _capture(monkeypatch)
    _stub(monkeypatch, {})

    assert cli.main(["--repo", str(repo), "--session", "s1"]) == 0

    assert "uncommitted" in stream.getvalue()


def test_main_repo_stays_quiet_on_a_clean_checkout(tmp_path: Path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    stream = _capture(monkeypatch)
    _stub(monkeypatch, {})

    assert cli.main(["--repo", str(repo), "--session", "s1"]) == 0

    assert "uncommitted" not in stream.getvalue()


def test_session_config_without_artifacts_keeps_the_old_layout(tmp_path: Path) -> None:
    config = cli._session_config(worktree=tmp_path, session="s1")

    assert "artifacts_dir" not in config["configurable"]
