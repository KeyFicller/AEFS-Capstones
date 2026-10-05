import io

from langchain_core.messages import HumanMessage
from repl_console import repl
from repl_console.commands import CommandOutcome, PromptCommand
from repl_console.repl import EndSessionError, Repl
from rich.console import Console


def _sink(monkeypatch) -> io.StringIO:
    stream = io.StringIO()
    monkeypatch.setattr(
        repl, "CONSOLE", Console(file=stream, width=100, no_color=True, force_terminal=False)
    )
    return stream


def test_render_error_survives_brackets(monkeypatch) -> None:
    stream = _sink(monkeypatch)
    repl.render_error("KeyError: 'a[b]c'")
    assert "a[b]c" in stream.getvalue()


def test_banner_survives_brackets(monkeypatch) -> None:
    stream = _sink(monkeypatch)
    repl.banner(title="repl", info={"worktree": "/tmp/[a]", "model": "m[b]"})
    text = stream.getvalue()
    assert "/tmp/[a]" in text and "m[b]" in text


def test_render_reply_handles_multimodal_content(monkeypatch) -> None:
    stream = _sink(monkeypatch)
    repl.render_reply(
        [{"type": "text", "text": "# title"}, {"type": "image_url", "image_url": {"url": "x"}}]
    )
    assert "title" in stream.getvalue()
    assert "[image]" in stream.getvalue()


def test_render_shell_shows_output_and_exit_code(monkeypatch) -> None:
    stream = _sink(monkeypatch)
    repl.render_shell(exit_code=0, output="hello\n")
    assert "hello" in stream.getvalue()
    assert "exit 0" in stream.getvalue()


def test_render_shell_survives_rich_markup(monkeypatch) -> None:
    stream = _sink(monkeypatch)
    repl.render_shell(exit_code=0, output="[red]boom[/red]")
    assert "[red]boom[/red]" in stream.getvalue()


def test_render_local_prints_plain_text(monkeypatch) -> None:
    stream = _sink(monkeypatch)
    repl.render_local("/help  list the registered commands")
    assert "/help" in stream.getvalue()


def test_run_shell_captures_output(tmp_path) -> None:
    exit_code, output = repl.run_shell("echo hi", tmp_path, timeout=5)
    assert exit_code == 0
    assert "hi" in output


def test_run_shell_reports_a_nonzero_exit(tmp_path) -> None:
    exit_code, _output = repl.run_shell("exit 3", tmp_path, timeout=5)
    assert exit_code == 3


def test_run_shell_uses_the_root_even_if_an_extra_dir_exists(tmp_path) -> None:
    root = tmp_path / "root"
    extra = tmp_path / "extra"
    root.mkdir()
    extra.mkdir()
    (extra / "marker.txt").write_text("nope", encoding="utf-8")
    exit_code, output = repl.run_shell("pwd", root, timeout=5)
    assert exit_code == 0
    assert str(root.resolve()) in output
    assert "extra" not in output


def _script(monkeypatch, lines: list[str]) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    pending = iter(lines)

    def _read(prompt: str = "") -> str:
        try:
            return next(pending)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr("builtins.input", _read)


def _repl(tmp_path, on_task, lines, monkeypatch, **kwargs) -> None:
    _script(monkeypatch, lines)
    Repl(title="t", info={"root": str(tmp_path)}, root=tmp_path, on_task=on_task, **kwargs).run()


def test_a_task_reaches_on_task(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    _repl(tmp_path, lambda message: seen.append(message.content), ["hello"], monkeypatch)
    assert seen == ["hello"]


def test_a_bang_line_does_not_reach_on_task(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    _repl(tmp_path, lambda message: seen.append(message.content), ["!echo hi"], monkeypatch)
    assert seen == []


def test_bang_does_not_run_in_an_extra_dir(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    extra = tmp_path / "extra"
    root.mkdir()
    extra.mkdir()
    seen: list[str] = []
    _script(monkeypatch, ["!pwd"])
    Repl(
        title="t",
        info={},
        root=root,
        extra_dirs=(extra,),
        on_task=lambda message: seen.append(message.content),
    ).run()
    assert seen == []


def test_an_unknown_command_does_not_reach_on_task(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    _repl(tmp_path, lambda message: seen.append(message.content), ["/nope"], monkeypatch)
    assert seen == []


def test_quit_ends_the_session(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    _repl(
        tmp_path,
        lambda message: seen.append(message.content),
        ["/quit", "still here"],
        monkeypatch,
    )
    assert seen == []


def test_a_prompt_command_is_not_parsed_again(tmp_path, monkeypatch) -> None:
    class _Review(PromptCommand):
        name = "review"
        summary = "review"

        def run(self, ctx, args):
            return CommandOutcome(prompt="look at @missing.txt")

    seen: list[HumanMessage] = []
    _repl(
        tmp_path,
        seen.append,
        ["/review x"],
        monkeypatch,
        commands=[_Review()],
    )
    assert seen[0].content == "look at @missing.txt"


def test_an_attachment_error_keeps_the_session_alive(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    _repl(
        tmp_path,
        lambda message: seen.append(message.content),
        ["look at @nope.txt", "still here"],
        monkeypatch,
    )
    assert seen == ["still here"]


def test_on_task_exception_does_not_end_the_session(tmp_path, monkeypatch) -> None:
    seen: list[str] = []

    def on_task(message: HumanMessage) -> None:
        if message.content == "boom":
            raise RuntimeError("nope")
        seen.append(str(message.content))

    _repl(tmp_path, on_task, ["boom", "after"], monkeypatch)
    assert seen == ["after"]


def test_end_session_stops_the_loop(tmp_path, monkeypatch) -> None:
    def on_task(message: HumanMessage) -> None:
        raise EndSessionError

    _repl(tmp_path, on_task, ["stop", "after"], monkeypatch)


def test_on_ready_runs_after_the_banner_and_before_input(tmp_path, monkeypatch) -> None:
    order: list[str] = []
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    def _read(prompt: str = "") -> str:
        order.append("read")
        raise EOFError

    monkeypatch.setattr("builtins.input", _read)
    Repl(
        title="t",
        info={},
        root=tmp_path,
        on_task=lambda message: None,
        on_ready=lambda: order.append("ready"),
    ).run()
    assert order == ["ready", "read"]


def test_blank_lines_are_ignored(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    _repl(tmp_path, lambda message: seen.append(message.content), ["", "   ", "go"], monkeypatch)
    assert seen == ["go"]
