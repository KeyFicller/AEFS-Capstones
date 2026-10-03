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

    def get_state(self, config):  # noqa: ARG002
        return SimpleNamespace(tasks=())

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


from langgraph.types import Command


class _EmptySnapshot:
    tasks: tuple = ()


class _PausingGraph:
    """First invoke pauses on an approval; a `Command` input means resume."""

    def __init__(self) -> None:
        self.log: list[str] = []
        self.payloads: list = []

    def update_state(self, config, values) -> None:  # noqa: ARG002
        self.log.append("reset")

    def invoke(self, payload, config) -> dict:  # noqa: ARG002
        self.log.append("invoke")
        self.payloads.append(payload)
        if isinstance(payload, Command):
            return {"messages": [], "turns": 1, "stop_reason": "completed"}
        pending = SimpleNamespace(value={"plan": "[-]  a"})
        return {"messages": [], "turns": 1, "__interrupt__": [pending]}

    def get_state(self, config):  # noqa: ARG002
        return _EmptySnapshot()


def test_session_config_enables_hitl(tmp_path: Path) -> None:
    config = cli._session_config(worktree=tmp_path, session="s1")

    assert config["configurable"]["enable_hitl"] is True


def test_repl_drains_a_paused_plan_through_approval(tmp_path, monkeypatch) -> None:
    _capture(monkeypatch)
    graph = _PausingGraph()
    monkeypatch.setattr(ui, "ask", _asker(["fix the typo"]))
    monkeypatch.setattr(ui, "ask_approval", lambda payload, **kwargs: "approve")

    cli.repl(
        graph=graph,
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="m",
    )

    assert graph.log == ["reset", "invoke", "invoke"]  # turn, then one resume
    assert isinstance(graph.payloads[-1], Command)


def test_repl_does_not_reprint_the_plan_the_turn_already_rendered(tmp_path, monkeypatch) -> None:
    """`_announce_todos` printed this plan as the turn's Tasks panel; printing it again doubles it."""
    _capture(monkeypatch)
    seen: list[dict] = []
    monkeypatch.setattr(ui, "ask", _asker(["fix the typo"]))
    monkeypatch.setattr(
        ui, "ask_approval", lambda payload, **kwargs: seen.append(kwargs) or "approve"
    )

    cli.repl(
        graph=_PausingGraph(),
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="m",
    )

    assert seen == [{"show_plan": False}]


def test_repl_prompts_for_approval_outside_the_live_region(tmp_path, monkeypatch) -> None:
    """The prompt must not fight the todo Live for the cursor (it is input(), not output)."""
    monkeypatch.setenv("TERM", "xterm-256color")  # else rich treats the console as dumb
    console = Console(file=io.StringIO(), width=80, force_terminal=True)
    panel = ui.TodoPanel(console)
    live_states: list[bool] = []

    def spy_approval(payload, **kwargs):  # noqa: ARG001
        live_states.append(panel._live is not None)
        return "approve"

    monkeypatch.setattr(ui, "ask_approval", spy_approval)
    monkeypatch.setattr(ui, "ask", _asker(["fix the typo"]))
    config = cli._session_config(worktree=tmp_path, session="s1")
    config["configurable"]["todo_renderer"] = panel

    cli.repl(graph=_PausingGraph(), config=config, model_name="m")

    assert live_states == [False], "approval ran while the Live region was still open"


class _LiveSpyGraph(_PausingGraph):
    """Records whether the todo Live was still open when the resume actually ran."""

    def __init__(self, panel) -> None:
        super().__init__()
        self.panel = panel
        self.live_at_resume: list[bool] = []

    def invoke(self, payload, config) -> dict:
        if isinstance(payload, Command):
            self.live_at_resume.append(self.panel._live is not None)
        return super().invoke(payload, config)


def test_repl_resumes_inside_the_live_region_so_the_panel_refreshes(tmp_path, monkeypatch) -> None:
    """Only the prompt must dodge the Live; the resumed graph still redraws in place."""
    monkeypatch.setenv("TERM", "xterm-256color")  # else rich treats the console as dumb
    console = Console(file=io.StringIO(), width=80, force_terminal=True)
    panel = ui.TodoPanel(console)
    graph = _LiveSpyGraph(panel)
    monkeypatch.setattr(ui, "ask_approval", lambda payload, **kwargs: "approve")
    monkeypatch.setattr(ui, "ask", _asker(["fix the typo"]))
    config = cli._session_config(worktree=tmp_path, session="s1")
    config["configurable"]["todo_renderer"] = panel

    cli.repl(graph=graph, config=config, model_name="m")

    assert graph.live_at_resume == [True], "the resumed half appended panels instead of redrawing"


class _PendingSnapshot:
    tasks = (SimpleNamespace(interrupts=(SimpleNamespace(value={"plan": "[-]  a"}),)),)


class _PendingAtStartupGraph(_PausingGraph):
    """A process killed while paused: the checkpoint still holds the interrupt."""

    def get_state(self, config):  # noqa: ARG002
        return _PendingSnapshot()


def test_repl_rebuilds_a_pending_approval_at_startup(tmp_path, monkeypatch) -> None:
    """A rebuilt approval has no panel above it, so this is the one path that prints the plan."""
    _capture(monkeypatch)
    graph = _PendingAtStartupGraph()
    seen: list[dict] = []
    monkeypatch.setattr(ui, "ask", _asker([]))  # EOF: no user turn at all
    monkeypatch.setattr(
        ui, "ask_approval", lambda payload, **kwargs: seen.append(kwargs) or "approve"
    )

    cli.repl(
        graph=graph,
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="m",
    )

    assert graph.log == ["invoke"]  # resumed before any input was read
    assert seen == [{"show_plan": True}]


def _unexpected(*_args, **_kwargs):
    raise AssertionError("the plan prompt must not see a question payload")


def _dismiss(*_args, **_kwargs):
    raise EOFError


def _interrupt(*_args, **_kwargs):
    raise KeyboardInterrupt


class _QuestionGraph(_PausingGraph):
    """First invoke pauses on an `ask_user` question; a `Command` input means resume."""

    def invoke(self, payload, config) -> dict:  # noqa: ARG002
        self.log.append("invoke")
        self.payloads.append(payload)
        if isinstance(payload, Command):
            return {"messages": [], "turns": 1, "stop_reason": "completed"}
        pending = SimpleNamespace(
            value={"type": "question", "question": "which?", "options": ["a", "b"]}
        )
        return {"messages": [], "turns": 1, "__interrupt__": [pending]}


def test_repl_answers_a_question_with_the_selection(tmp_path, monkeypatch) -> None:
    _capture(monkeypatch)
    graph = _QuestionGraph()
    monkeypatch.setattr(ui, "ask", _asker(["fix the typo"]))
    monkeypatch.setattr(ui, "ask_question", lambda payload: {"answer": "b", "cancelled": False})
    monkeypatch.setattr(ui, "ask_approval", _unexpected)

    cli.repl(
        graph=graph,
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="m",
    )

    assert graph.payloads[-1].resume == {"answer": "b", "cancelled": False}


def test_repl_maps_a_dismissed_question_to_the_cancelled_sentinel(tmp_path, monkeypatch) -> None:
    """A question is not a plan gate: Ctrl-C lets the agent carry on, it does not reject."""
    _capture(monkeypatch)
    graph = _QuestionGraph()
    monkeypatch.setattr(ui, "ask", _asker(["fix the typo"]))
    monkeypatch.setattr(ui, "ask_question", _interrupt)

    cli.repl(
        graph=graph,
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="m",
    )

    assert graph.payloads[-1].resume == {"answer": None, "cancelled": True}


def test_repl_ends_the_turn_on_eof_at_a_question(tmp_path, monkeypatch) -> None:
    """EOF closes the answer channel for good: resuming would ask again forever."""
    _capture(monkeypatch)
    graph = _QuestionGraph()
    monkeypatch.setattr(ui, "ask", _asker(["fix the typo"]))
    monkeypatch.setattr(ui, "ask_question", _dismiss)

    cli.repl(
        graph=graph,
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="m",
    )

    # Only the task turn ran; no resume was issued with a sentinel answer.
    assert all(isinstance(payload, dict) for payload in graph.payloads)


def test_repl_keeps_rejecting_a_declined_plan_prompt(tmp_path, monkeypatch) -> None:
    """The plan gate's dismissal semantics must not follow the question's."""
    _capture(monkeypatch)
    graph = _PausingGraph()
    monkeypatch.setattr(ui, "ask", _asker(["fix the typo"]))
    monkeypatch.setattr(ui, "ask_approval", _dismiss)

    cli.repl(
        graph=graph,
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="m",
    )

    assert graph.payloads[-1].resume == "reject"
