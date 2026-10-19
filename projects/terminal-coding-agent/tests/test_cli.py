import io
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.types import Command
from repl_console.commands import CommandContext, CommandOutcome, PromptCommand
from rich.console import Console
from terminal_coding_agent import cli, ui
from terminal_coding_agent.tools import display


def _capture(monkeypatch) -> io.StringIO:
    stream = io.StringIO()
    monkeypatch.setattr(
        ui, "CONSOLE", Console(file=stream, width=100, no_color=True, force_terminal=False)
    )
    return stream


@pytest.fixture(autouse=True)
def _restore_display_level(monkeypatch) -> None:
    """`main` sets the process-wide tool-output level; keep it out of the other test files."""
    monkeypatch.setattr(display, "LEVEL", display.LEVEL)


def test_tool_output_command_sets_the_level(monkeypatch) -> None:
    monkeypatch.setattr(display, "LEVEL", "detail")
    command = cli.ToolOutputCommand()
    ctx = CommandContext(root=Path("/tmp"))

    assert command.run(ctx, "").message.startswith("tool output detail")
    assert display.LEVEL == "detail", "a bare invocation only reports"

    assert command.run(ctx, " simple ").message == "tool output simple"
    assert display.LEVEL == "simple"

    assert command.run(ctx, "off").message == "tool output off"
    assert display.LEVEL == "off"


def test_tool_output_command_ignores_a_bad_argument(monkeypatch) -> None:
    monkeypatch.setattr(display, "LEVEL", "off")
    command = cli.ToolOutputCommand()

    message = command.run(CommandContext(root=Path("/tmp")), "on").message

    assert "tool output off" in message and "usage" in message, "the old on/off words are gone"
    assert display.LEVEL == "off"


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
    assert called["config"]["configurable"]["shell_anything"] is False


def test_main_shell_anything_sets_the_flag(tmp_path: Path, monkeypatch) -> None:
    called: dict = {}
    monkeypatch.setattr(cli, "load_local_env", lambda path: None)
    monkeypatch.setattr(cli, "make_graph", lambda config: _FakeGraph())
    monkeypatch.setattr(cli, "build_models", lambda config: SimpleNamespace(planner="stub"))
    monkeypatch.setattr(cli, "resolve_model_name", lambda model: "stub-model")
    monkeypatch.setattr(cli, "repl", lambda **kwargs: called.update(kwargs))

    assert cli.main(["--worktree", str(tmp_path), "--shell_anything"]) == 0
    assert called["config"]["configurable"]["shell_anything"] is True


def test_main_defaults_to_a_persistent_workspace(tmp_path: Path, monkeypatch) -> None:
    """Without --worktree the agent writes under .agent-workspaces and the edits survive."""
    seen: dict = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_local_env", lambda path: None)
    monkeypatch.setattr(cli, "make_graph", lambda config: _FakeGraph())
    monkeypatch.setattr(cli, "build_models", lambda config: SimpleNamespace(planner="stub"))
    monkeypatch.setattr(cli, "resolve_model_name", lambda model: "stub-model")

    def fake_repl(**kwargs) -> None:
        seen.update(kwargs)
        worktree = Path(kwargs["config"]["configurable"]["worktree"])
        assert worktree.is_dir(), "the worktree must exist while the session runs"
        (worktree / "notes.txt").write_text("kept", encoding="utf-8")

    monkeypatch.setattr(cli, "repl", fake_repl)

    assert cli.main(["--session", "s1"]) == 0

    worktree = Path(seen["config"]["configurable"]["worktree"])
    assert worktree == tmp_path / cli.WORKSPACES_DIRNAME / "s1"
    assert (worktree / "notes.txt").read_text(encoding="utf-8") == "kept"


class _FakeSandbox:
    instances: list = []

    def __init__(
        self, *, worktree: Path, image: str, session: str, allow_dangerous: bool = False
    ) -> None:
        self.worktree = worktree
        self.image = image
        self.session = session
        self.allow_dangerous = allow_dangerous
        self.log: list[str] = []
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.log.append("start")

    def close(self) -> None:
        self.log.append("close")


def _stub_main(monkeypatch, called: dict) -> None:
    monkeypatch.setattr(cli, "load_local_env", lambda path: None)
    monkeypatch.setattr(cli, "make_graph", lambda config: _FakeGraph())
    monkeypatch.setattr(cli, "build_models", lambda config: SimpleNamespace(planner="stub"))
    monkeypatch.setattr(cli, "resolve_model_name", lambda model: "stub-model")
    monkeypatch.setattr(cli, "repl", lambda **kwargs: called.update(kwargs))


def test_main_docker_starts_a_sandbox_and_closes_it(tmp_path: Path, monkeypatch) -> None:
    called: dict = {}
    _stub_main(monkeypatch, called)
    _FakeSandbox.instances = []
    monkeypatch.setattr(cli, "require_docker", lambda: None)
    monkeypatch.setattr(cli, "DockerSandbox", _FakeSandbox)

    assert cli.main(["--worktree", str(tmp_path), "--session", "s1", "--docker"]) == 0

    box = _FakeSandbox.instances[0]
    assert (box.worktree, box.image, box.session) == (tmp_path, "python:3.13-slim", "s1")
    assert box.log == ["start", "close"]
    assert called["config"]["configurable"]["sandbox"] is box


def test_main_docker_image_is_configurable(tmp_path: Path, monkeypatch) -> None:
    called: dict = {}
    _stub_main(monkeypatch, called)
    _FakeSandbox.instances = []
    monkeypatch.setattr(cli, "require_docker", lambda: None)
    monkeypatch.setattr(cli, "DockerSandbox", _FakeSandbox)

    assert cli.main(["--worktree", str(tmp_path), "--docker", "--docker-image", "img:2"]) == 0
    assert _FakeSandbox.instances[0].image == "img:2"


def test_main_without_docker_has_no_sandbox(tmp_path: Path, monkeypatch) -> None:
    called: dict = {}
    _stub_main(monkeypatch, called)

    assert cli.main(["--worktree", str(tmp_path)]) == 0
    assert called["config"]["configurable"]["sandbox"] is None


def test_main_fails_fast_when_docker_is_missing(tmp_path: Path, monkeypatch) -> None:
    called: dict = {}
    _stub_main(monkeypatch, called)
    monkeypatch.setattr(
        cli, "require_docker", lambda: (_ for _ in ()).throw(cli.SandboxError("no daemon"))
    )

    assert cli.main(["--worktree", str(tmp_path), "--docker"]) == 1
    assert called == {}, "the REPL must not open without a working sandbox"


def test_main_rejects_dangerous_without_docker(tmp_path: Path, monkeypatch) -> None:
    """No container means no fence: refuse instead of running the agent on the host."""
    called: dict = {}
    _stub_main(monkeypatch, called)

    assert cli.main(["--worktree", str(tmp_path), "--allow-dangerous"]) == 2
    assert called == {}, "the REPL must not open when the flag is misused"


def test_main_dangerous_opens_the_sandbox(tmp_path: Path, monkeypatch) -> None:
    called: dict = {}
    _stub_main(monkeypatch, called)
    _capture(monkeypatch)
    _FakeSandbox.instances = []
    monkeypatch.setattr(cli, "require_docker", lambda: None)
    monkeypatch.setattr(cli, "DockerSandbox", _FakeSandbox)

    assert cli.main(["--worktree", str(tmp_path), "--docker", "--allow-dangerous"]) == 0

    assert _FakeSandbox.instances[0].allow_dangerous is True
    assert called["config"]["configurable"]["allow_dangerous"] is True


def test_main_is_fenced_by_default(tmp_path: Path, monkeypatch) -> None:
    called: dict = {}
    _stub_main(monkeypatch, called)
    monkeypatch.setattr(cli, "require_docker", lambda: None)
    monkeypatch.setattr(cli, "DockerSandbox", _FakeSandbox)
    _FakeSandbox.instances = []

    assert cli.main(["--worktree", str(tmp_path), "--docker"]) == 0

    assert _FakeSandbox.instances[0].allow_dangerous is False
    assert called["config"]["configurable"]["allow_dangerous"] is False


class _EmptySnapshot:
    tasks: tuple = ()


class _QuestionGraph:
    """First invoke pauses on an `ask_user` question; a `Command` input means resume."""

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
        pending = SimpleNamespace(
            value={"type": "question", "question": "which?", "options": ["a", "b"]}
        )
        return {"messages": [], "turns": 1, "__interrupt__": [pending]}

    def get_state(self, config):  # noqa: ARG002
        return _EmptySnapshot()


def _pending_at_startup(config):  # noqa: ARG001
    """A process killed while paused: the checkpoint still holds the interrupt."""
    return SimpleNamespace(
        tasks=(
            SimpleNamespace(
                interrupts=(
                    SimpleNamespace(
                        value={"type": "question", "question": "which?", "options": ["a"]}
                    ),
                )
            ),
        )
    )


def test_repl_pauses_then_resumes_a_question(tmp_path, monkeypatch) -> None:
    _capture(monkeypatch)
    graph = _QuestionGraph()
    _script(monkeypatch, ["fix the typo"])
    monkeypatch.setattr(ui, "ask_question", lambda payload: {"answer": "b", "cancelled": False})

    cli.repl(
        graph=graph,
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="m",
    )

    assert graph.log == ["reset", "invoke", "invoke"]  # turn, then one resume
    assert isinstance(graph.payloads[-1], Command)


def test_repl_resumes_inside_the_live_region_so_the_panel_refreshes(tmp_path, monkeypatch) -> None:
    """Only the prompt must dodge the Live; the resumed graph still redraws in place."""
    monkeypatch.setenv("TERM", "xterm-256color")  # else rich treats the console as dumb
    console = Console(file=io.StringIO(), width=80, force_terminal=True)
    panel = ui.TodoPanel(console)
    live_at_resume: list[bool] = []

    class LiveSpyGraph(_QuestionGraph):
        def invoke(self, payload, config):
            if isinstance(payload, Command):
                live_at_resume.append(panel._live is not None)
            return super().invoke(payload, config)

    graph = LiveSpyGraph()
    monkeypatch.setattr(ui, "ask_question", lambda payload: {"answer": "b", "cancelled": False})
    _script(monkeypatch, ["fix the typo"])
    config = cli._session_config(worktree=tmp_path, session="s1")
    config["configurable"]["todo_renderer"] = panel

    cli.repl(graph=graph, config=config, model_name="m")

    assert live_at_resume == [True], "the resumed half appended panels instead of redrawing"


def test_repl_rebuilds_a_pending_question_at_startup(tmp_path, monkeypatch) -> None:
    """A process killed while paused resumes before any input is read."""
    _capture(monkeypatch)
    graph = _QuestionGraph()
    monkeypatch.setattr(graph, "get_state", _pending_at_startup)
    _script(monkeypatch, [])  # EOF: no user turn at all
    monkeypatch.setattr(ui, "ask_question", lambda payload: {"answer": "b", "cancelled": False})

    cli.repl(
        graph=graph,
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="m",
    )

    assert graph.log == ["invoke"]  # resumed before any input was read


def _dismiss(*_args, **_kwargs):
    raise EOFError


def _interrupt(*_args, **_kwargs):
    raise KeyboardInterrupt


def test_repl_answers_a_question_with_the_selection(tmp_path, monkeypatch) -> None:
    _capture(monkeypatch)
    graph = _QuestionGraph()
    _script(monkeypatch, ["fix the typo"])
    monkeypatch.setattr(ui, "ask_question", lambda payload: {"answer": "b", "cancelled": False})

    cli.repl(
        graph=graph,
        config=cli._session_config(worktree=tmp_path, session="s1"),
        model_name="m",
    )

    assert graph.payloads[-1].resume == {"answer": "b", "cancelled": False}


def test_repl_maps_a_dismissed_question_to_the_cancelled_sentinel(tmp_path, monkeypatch) -> None:
    """Ctrl-C lets the agent carry on; it does not reject the turn."""
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
