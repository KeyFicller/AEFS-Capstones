from __future__ import annotations

import io
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from terminal_coding_agent import cli, ui


def _capture(monkeypatch) -> io.StringIO:
    stream = io.StringIO()
    monkeypatch.setattr(
        ui, "CONSOLE", Console(file=stream, width=100, no_color=True, force_terminal=False)
    )
    return stream


def _asker(lines: list[str]):
    pending = iter(lines)

    def ask() -> str:
        try:
            return next(pending)
        except StopIteration as exc:  # a real prompt raises EOFError on Ctrl-D
            raise EOFError from exc

    return ask


class _FakeGraph:
    def __init__(self) -> None:
        self.log: list[str] = []

    def update_state(self, config, values) -> None:  # noqa: ARG002
        self.log.append("reset")

    def invoke(self, payload, config) -> dict:  # noqa: ARG002
        self.log.append("invoke")
        return {"messages": [], "turns": 1, "tokens": 10, "cost_rmb": 0.0, "stop_reason": None}


class _RecordingTodos:
    """Stands in for `ui.TodoPanel`: records the region around each turn."""

    def __init__(self) -> None:
        self.log: list[str] = []

    def __call__(self, text: str, version: int = 0) -> None:  # noqa: ARG002
        self.log.append("update")

    def region(self):
        @contextmanager
        def opened():
            self.log.append("region-open")
            try:
                yield
            finally:
                self.log.append("region-close")

        return opened()


def test_session_config_injects_the_renderers(tmp_path: Path, monkeypatch) -> None:
    _capture(monkeypatch)

    config = cli._session_config(worktree=tmp_path, session="s1")

    assert config["configurable"]["thread_id"] == "s1"
    assert config["configurable"]["worktree"] == str(tmp_path)
    assert isinstance(config["configurable"]["todo_renderer"], ui.TodoPanel)
    assert isinstance(config["configurable"]["tool_renderer"], ui.ToolLog)


def test_repl_opens_a_todo_region_around_each_turn(tmp_path: Path, monkeypatch) -> None:
    """The renderer must be live for the whole turn, or panels append instead of refresh."""
    _capture(monkeypatch)
    graph = _FakeGraph()
    config = cli._session_config(worktree=tmp_path, session="s1")
    todos = _RecordingTodos()
    config["configurable"]["todo_renderer"] = todos
    monkeypatch.setattr(ui, "ask", _asker(["fix the typo"]))

    cli.repl(graph=graph, config=config, model_name="deepseek:deepseek-v4-flash")

    assert graph.log == ["reset", "invoke"]
    assert todos.log == ["region-open", "region-close"]


def test_repl_runs_every_turn_through_the_graph(tmp_path: Path, monkeypatch) -> None:
    _capture(monkeypatch)
    graph = _FakeGraph()
    monkeypatch.setattr(ui, "ask", _asker(["fix the typo"]))

    cli.repl(
        graph=graph,
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="deepseek:deepseek-v4-flash",
    )

    assert graph.log == ["reset", "invoke"]


def test_repl_answers_eof_by_returning(tmp_path: Path, monkeypatch) -> None:
    _capture(monkeypatch)
    monkeypatch.setattr(ui, "ask", _asker([]))

    cli.repl(
        graph=_FakeGraph(),
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="deepseek:deepseek-v4-flash",
    )


def test_main_goes_straight_to_the_repl(tmp_path: Path, monkeypatch) -> None:
    """No subcommand: the REPL is the only mode, built from --worktree / --session."""
    called: dict = {}
    monkeypatch.setattr(cli, "load_local_env", lambda path: None)
    monkeypatch.setattr(cli, "make_graph", lambda config: _FakeGraph())
    monkeypatch.setattr(cli, "build_models", lambda config: SimpleNamespace(planner="stub"))
    monkeypatch.setattr(cli, "resolve_model_name", lambda model: "stub-model")

    def fake_repl(**kwargs) -> None:
        called.update(kwargs)

    monkeypatch.setattr(cli, "repl", fake_repl)

    assert cli.main(["--worktree", str(tmp_path), "--session", "s1"]) == 0
    assert called["model_name"] == "stub-model"
    assert called["config"]["configurable"]["thread_id"] == "s1"
    assert called["config"]["configurable"]["worktree"] == str(tmp_path)


def test_main_defaults_to_a_throwaway_worktree(monkeypatch) -> None:
    """Without --worktree the agent must not touch the repo, so .agent/ lands in a temp dir."""
    seen: dict = {}
    monkeypatch.setattr(cli, "load_local_env", lambda path: None)
    monkeypatch.setattr(cli, "make_graph", lambda config: _FakeGraph())
    monkeypatch.setattr(cli, "build_models", lambda config: SimpleNamespace(planner="stub"))
    monkeypatch.setattr(cli, "resolve_model_name", lambda model: "stub-model")

    def fake_repl(**kwargs) -> None:
        seen.update(kwargs)
        worktree = Path(kwargs["config"]["configurable"]["worktree"])
        assert worktree.is_dir(), "the worktree must exist while the session runs"
        (worktree / ".agent").mkdir()  # what a real turn would create

    monkeypatch.setattr(cli, "repl", fake_repl)

    assert cli.main([]) == 0

    worktree = Path(seen["config"]["configurable"]["worktree"])
    assert worktree != Path.cwd()
    assert worktree.is_relative_to(Path(tempfile.gettempdir()))
    assert not worktree.exists(), "a throwaway worktree must not outlive the session"
