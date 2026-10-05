import io
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from langgraph.types import Command
from repl_console.commands import CommandOutcome, PromptCommand
from rich.console import Console
from terminal_coding_agent import cli, ui


def _capture(monkeypatch) -> io.StringIO:
    stream = io.StringIO()
    monkeypatch.setattr(
        ui, "CONSOLE", Console(file=stream, width=100, no_color=True, force_terminal=False)
    )
    return stream


def _script(monkeypatch, lines: list[str]) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    pending = iter(lines)

    def _read(prompt: str = "") -> str:
        try:
            return next(pending)
        except StopIteration as exc:  # a real prompt raises EOFError on Ctrl-D
            raise EOFError from exc

    monkeypatch.setattr("builtins.input", _read)


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


def test_session_config_streams_the_summary_only_on_a_terminal(tmp_path: Path, monkeypatch) -> None:
    """Harbor logs and pipes have no Live to redraw: they keep the one-shot reply panel."""
    _capture(monkeypatch)

    config = cli._session_config(worktree=tmp_path, session="s1")

    assert "summary_renderer" not in config["configurable"]


def test_session_config_shares_the_panel_between_todos_and_the_summary(tmp_path, monkeypatch) -> None:
    """One instance for both, or two redrawers fight over the cursor."""
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(ui, "CONSOLE", Console(file=io.StringIO(), width=80, force_terminal=True))

    configurable = cli._session_config(worktree=tmp_path, session="s1")["configurable"]

    assert configurable["summary_renderer"] == configurable["todo_renderer"].stream_summary


def test_render_task_result_skips_a_streamed_reply_but_keeps_the_footer(monkeypatch) -> None:
    stream = _capture(monkeypatch)

    cli._render_task_result(
        {
            "messages": [SimpleNamespace(content="already on screen")],
            "turns": 2,
            "stop_reason": "completed",
        },
        show_reply=False,
    )

    out = stream.getvalue()
    assert "already on screen" not in out
    assert "turns" in out and "completed" in out


def test_repl_opens_a_todo_region_around_each_turn(tmp_path: Path, monkeypatch) -> None:
    """The renderer must be live for the whole turn, or panels append instead of refresh."""
    _capture(monkeypatch)
    graph = _FakeGraph()
    config = cli._session_config(worktree=tmp_path, session="s1")
    todos = _RecordingTodos()
    config["configurable"]["todo_renderer"] = todos
    _script(monkeypatch, ["fix the typo"])

    cli.repl(graph=graph, config=config, model_name="deepseek:deepseek-v4-flash")

    assert graph.log == ["reset", "invoke"]
    assert todos.log == ["region-open", "region-close"]


def test_repl_runs_every_turn_through_the_graph(tmp_path: Path, monkeypatch) -> None:
    _capture(monkeypatch)
    graph = _FakeGraph()
    _script(monkeypatch, ["fix the typo"])

    cli.repl(
        graph=graph,
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="deepseek:deepseek-v4-flash",
    )

    assert graph.log == ["reset", "invoke"]


def test_repl_answers_eof_by_returning(tmp_path: Path, monkeypatch) -> None:
    _capture(monkeypatch)
    _script(monkeypatch, [])

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
    _script(monkeypatch, ["fix the typo"])
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
    _script(monkeypatch, ["fix the typo"])
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
    _script(monkeypatch, ["fix the typo"])
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
    _script(monkeypatch, ["fix the typo"])
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
    _script(monkeypatch, [])  # EOF: no user turn at all
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
    _script(monkeypatch, ["fix the typo"])
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
    _script(monkeypatch, ["fix the typo"])
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
    _script(monkeypatch, ["fix the typo"])
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
    _script(monkeypatch, ["fix the typo"])
    monkeypatch.setattr(ui, "ask_approval", _dismiss)

    cli.repl(
        graph=graph,
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="m",
    )

    assert graph.payloads[-1].resume == "reject"


def test_session_config_enables_answer_mode(tmp_path: Path) -> None:
    config = cli._session_config(worktree=tmp_path, session="s1")

    assert config["configurable"]["enable_answer_mode"] is True


class _CapturingGraph(_FakeGraph):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[dict] = []

    def invoke(self, payload, config):
        self.payloads.append(payload)
        return super().invoke(payload, config)


def _drive(monkeypatch, tmp_path, lines, extra_commands=()):
    _capture(monkeypatch)
    _script(monkeypatch, lines)
    graph = _CapturingGraph()
    cli.repl(
        graph=graph,
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="m",
        commands=extra_commands,
    )
    return graph


def test_a_bang_line_never_reaches_the_graph(tmp_path, monkeypatch) -> None:
    graph = _drive(monkeypatch, tmp_path, ["!echo hi"])
    assert graph.payloads == []


def test_an_unknown_slash_command_never_reaches_the_graph(tmp_path, monkeypatch) -> None:
    graph = _drive(monkeypatch, tmp_path, ["/nope"])
    assert graph.payloads == []


def test_slash_quit_ends_the_session_without_calling_the_graph(tmp_path, monkeypatch) -> None:
    graph = _drive(monkeypatch, tmp_path, ["/quit"])
    assert graph.payloads == []


def test_a_task_goes_to_the_graph_with_its_attachment(tmp_path, monkeypatch) -> None:
    (tmp_path / "NOTES.md").write_text("hello", encoding="utf-8")
    graph = _drive(monkeypatch, tmp_path, ["summarize @NOTES.md"])
    assert "[NOTES.md]\nhello" in graph.payloads[0]["messages"][0].content


def test_a_prompt_command_goes_to_the_graph_as_a_task(tmp_path, monkeypatch) -> None:
    class _Review(PromptCommand):
        name = "review"
        summary = "review a file"

        def run(self, ctx, args):
            return CommandOutcome(prompt=f"Review {args}.")

    graph = _drive(monkeypatch, tmp_path, ["/review src/a.py"], extra_commands=[_Review()])
    assert graph.payloads[0]["messages"][0].content == "Review src/a.py."


def test_an_attachment_error_keeps_the_session_alive(tmp_path, monkeypatch) -> None:
    graph = _drive(monkeypatch, tmp_path, ["look at @nope.txt", "!echo still here"])
    assert graph.payloads == []


def test_scripted_session_mixes_shorthands_and_tasks(tmp_path, monkeypatch) -> None:
    """`!` and `/help` never call the graph; a task does; `/quit` ends the loop."""
    (tmp_path / "NOTES.md").write_text("hello", encoding="utf-8")
    rendered: list[str] = []
    _capture(monkeypatch)
    monkeypatch.setattr("repl_console.repl.render_local", rendered.append)
    _script(monkeypatch, ["!echo hi", "/help", "summarize @NOTES.md", "/quit"])
    graph = _CapturingGraph()

    cli.repl(
        graph=graph,
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="m",
    )

    assert len(graph.payloads) == 1
    assert "[NOTES.md]\nhello" in graph.payloads[0]["messages"][0].content
    assert any("/help" in line for line in rendered)
