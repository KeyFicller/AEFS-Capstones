import importlib.util
import io
import os
import subprocess
import sys
import threading
import time

import pytest
from rich.console import Console
from terminal_coding_agent import ui
from terminal_coding_agent.state import ToDoItem, ToDoStatus, format_todos

SPINNER_FRAMES = set("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")


def _capture(monkeypatch) -> io.StringIO:
    stream = io.StringIO()
    monkeypatch.setattr(
        ui, "CONSOLE", Console(file=stream, width=100, no_color=True, force_terminal=False)
    )
    return stream


def _states() -> list[str]:
    return [
        format_todos([ToDoItem(status=ToDoStatus.PENDING, description="wire repl")]),
        format_todos([ToDoItem(status=ToDoStatus.IN_PROGRESS, description="wire repl")]),
        format_todos([ToDoItem(status=ToDoStatus.DONE, description="wire repl")]),
    ]


def test_panel_prints_each_state_outside_a_region(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    panel = ui.TodoPanel()

    for state in _states():
        panel(state)

    out = stream.getvalue()
    assert out.count("wire repl") == 3


def test_region_keeps_printing_each_state_off_a_terminal(monkeypatch) -> None:
    """Harbor logs and pipes have nothing to overwrite, so no state may be lost."""
    stream = _capture(monkeypatch)
    panel = ui.TodoPanel()

    with panel.region():
        for state in _states():
            panel(state)

    assert stream.getvalue().count("wire repl") == 3


def test_panel_is_titled_tasks_with_no_ascii_separators(monkeypatch) -> None:
    stream = _capture(monkeypatch)

    ui.TodoPanel()("[✓] wire the repl", 0)

    out = stream.getvalue()
    assert ui.TODO_TITLE in out
    assert "wire the repl" in out
    assert "#" not in out, "the box is drawn by rich now; the '####' chrome is gone"


def test_panel_shows_the_replan_count_as_a_subtitle(monkeypatch) -> None:
    stream = _capture(monkeypatch)

    ui.TodoPanel()("[✓] wire the repl", 2)

    assert "replan v2" in stream.getvalue()


def test_panel_always_shows_the_replan_count(monkeypatch) -> None:
    """It resets every turn, so hiding v0 would make the count effectively invisible."""
    stream = _capture(monkeypatch)

    ui.TodoPanel()("[✓] wire the repl", 0)

    assert "replan v0" in stream.getvalue()


def test_tool_log_summarises_each_call(monkeypatch) -> None:
    """One line per call: what ran, on what, and how it went."""
    stream = _capture(monkeypatch)
    log = ui.ToolLog()

    log("read_file", {"path": "src/a.py"}, "l1\nl2\nl3")
    log("ripgrep", {"pattern": "TODO"}, "a.py:1:x\nb.py:2:y")
    log("run_shell", {"command": "pytest -q"}, "exit_code: 0\nstdout:\nok\nstderr:\n")
    log("git", {"git_args": ["status", "--short"]}, "exit_code: 1\nstdout:\nstderr:\n")

    out = stream.getvalue()
    assert "read_file" in out and "src/a.py" in out and "3 lines" in out
    assert "2 hits" in out
    assert "pytest -q" in out and "exit 0" in out
    assert "status --short" in out and "exit 1" in out


def test_tool_log_surfaces_errors_and_empty_search(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    log = ui.ToolLog()

    log("read_file", {"path": "x"}, "Error: path escapes worktree: ../x")
    log("ripgrep", {"pattern": "zzz"}, "")
    log("ripgrep", {"pattern": "one"}, "a.py:1:x")

    out = stream.getvalue()
    assert "path escapes worktree" in out
    assert "no hits" in out
    assert "(1 hit)" in out, "a single hit must not read as '1 hits'"


def test_tool_log_renders_the_edit_diff(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    diff = "--- a.py\n+++ a.py\n@@ -1 +1 @@\n-old\n+new"

    ui.ToolLog()("edit_file", {"path": "a.py"}, diff)

    out = stream.getvalue()
    assert "a.py" in out and "+1 -1" in out
    assert "-old" in out and "+new" in out, "the diff body must be shown, not just its size"


def test_tool_log_skips_the_diff_when_the_edit_failed(monkeypatch) -> None:
    stream = _capture(monkeypatch)

    ui.ToolLog()("edit_file", {"path": "a.py"}, "Error: old_str not found in a.py")

    out = stream.getvalue()
    assert "old_str not found" in out
    assert "@@" not in out


def test_tool_log_keeps_long_arguments_on_one_line(monkeypatch) -> None:
    stream = _capture(monkeypatch)

    ui.ToolLog()("run_shell", {"command": "echo " + "x" * 200}, "exit_code: 0\nstdout:\n")

    line = next(line for line in stream.getvalue().splitlines() if "run_shell" in line)
    assert len(line) <= ui.MAX_TARGET_CHARS + 40
    assert line.count("\n") == 0


def test_render_budget_reports_turns_and_stop_reason(monkeypatch) -> None:
    """Token/cost totals must stay off the footer: the accounting is not trustworthy."""
    stream = _capture(monkeypatch)

    ui.render_budget({"turns": 3, "tokens": 1200, "cost_rmb": 0.5, "stop_reason": "max_turns"})

    out = stream.getvalue()
    assert "turns" in out and "3" in out
    assert "max_turns" in out
    assert "tokens" not in out and "1200" not in out
    assert "cost" not in out and "¥" not in out


def test_importing_ui_enables_a_line_editor() -> None:
    """Without readline, input() erases one byte per backspace and mangles CJK.

    Checked in a subprocess because another test may have imported readline
    already, which would mask a regression here.
    """
    if importlib.util.find_spec("readline") is None:
        pytest.skip("readline is unavailable on this platform")

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, terminal_coding_agent.ui; print('readline' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "True"


def _open_tty(monkeypatch):
    """A pty with a drain thread; returns (master, stream, chunks, reader)."""
    pty = pytest.importorskip("pty")
    monkeypatch.setenv("TERM", "xterm-256color")  # else rich treats the pty as dumb
    try:
        master, slave = pty.openpty()
    except OSError as exc:  # a sandbox that hides /dev/pty* is an environment limit
        pytest.skip(f"no pty available: {exc}")

    stream = os.fdopen(slave, "w")
    chunks: list[bytes] = []

    # The pty buffer is ~1KB, so a reader must drain it while the panel is drawn
    # or the writer blocks. A real terminal always reads; this stands in for it.
    def drain() -> None:
        while True:
            try:
                data = os.read(master, 65536)
            except OSError:
                return
            if not data:
                return
            chunks.append(data)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    return master, stream, chunks, reader


def test_region_redraws_in_place_on_a_terminal(monkeypatch) -> None:
    """The whole point: a status change must move the cursor back up, not append."""
    master, stream, chunks, reader = _open_tty(monkeypatch)
    panel = ui.TodoPanel(Console(file=stream, width=40))

    with panel.region():
        for state in _states():
            panel(state)
    stream.close()  # EOF ends the reader
    reader.join(timeout=5)
    os.close(master)

    written = b"".join(chunks)
    cursor_up = written.count(b"\x1b[1A") + written.count(b"\x1b[2A")
    assert cursor_up >= 2, f"expected in-place redraw, got {cursor_up} cursor-up sequences"


def test_a_turn_shows_an_animating_spinner_while_the_model_thinks(monkeypatch) -> None:
    """A long model call with no todo yet must not look like a hung terminal."""
    master, stream, chunks, reader = _open_tty(monkeypatch)
    panel = ui.TodoPanel(Console(file=stream, width=40))

    with panel.region():
        time.sleep(0.5)  # several auto-refresh ticks at refresh_per_second=10
        stream.flush()
        time.sleep(0.15)  # let the reader thread catch up
        during = b"".join(chunks).decode("utf-8", "replace")

    stream.close()
    reader.join(timeout=5)
    os.close(master)

    assert ui.WORKING_MESSAGE in during, "no spinner while the agent was working"
    frames = {char for char in during if char in SPINNER_FRAMES}
    assert len(frames) >= 2, f"spinner did not animate, frames seen: {frames}"


def test_spinner_is_rendered_only_while_working() -> None:
    """The spinner must not survive into the finished turn's panel."""
    panel = ui.TodoPanel()
    panel("[✓] wire the repl")

    panel._working = True
    assert ui.WORKING_MESSAGE in _render_to_text(panel._renderable())

    panel._working = False
    assert ui.WORKING_MESSAGE not in _render_to_text(panel._renderable())


def _screen_of(raw: bytes, *, cols: int, rows: int) -> str:
    """Replay terminal bytes into a fresh screen to read the layout a user would see."""
    pyte = pytest.importorskip("pyte", reason="layout assertions need a terminal emulator")
    screen = pyte.Screen(cols, rows)
    pyte.Stream(screen).feed(raw.decode("utf-8", "replace"))
    return "\n".join(line.rstrip() for line in screen.display).rstrip("\n")


def test_tool_lines_scroll_above_the_pinned_todo_panel(monkeypatch) -> None:
    """The CLI contract: question, then tool output, with the Tasks panel pinned last."""
    master, stream, chunks, reader = _open_tty(monkeypatch)
    console = Console(file=stream, width=48)
    todos = ui.TodoPanel(console)
    tools = ui.ToolLog(console)

    console.print("you > fix the typo")
    with todos.region():
        todos("[-] fix the typo", 0)
        tools("read_file", {"path": "src/a.py"}, "l1\nl2")
        tools("edit_file", {"path": "src/a.py"}, "--- a.py\n+++ a.py\n@@ -1 +1 @@\n-old\n+new")
        todos("[✓] fix the typo", 0)

    stream.close()
    reader.join(timeout=5)
    os.close(master)

    screen = _screen_of(b"".join(chunks), cols=48, rows=24)
    assert screen.count("Tasks") == 1, f"panel must stay pinned, not stack:\n{screen}"
    assert screen.index("you >") < screen.index("read_file") < screen.index("Tasks")
    assert "edit_file" in screen


def _render_to_text(renderable, *, width: int = 60) -> str:
    stream = io.StringIO()
    Console(file=stream, width=width, no_color=True).print(renderable)
    return stream.getvalue()


def test_ask_approval_accepts_y(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda *a: "y")

    assert ui.ask_approval({"plan": "[-]  step-a"}) == "approve"
    assert "step-a" in stream.getvalue()


def test_ask_approval_accepts_n(monkeypatch) -> None:
    _capture(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda *a: "n")

    assert ui.ask_approval({"plan": "x"}) == "reject"


def test_ask_approval_reasks_on_garbage(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    answers = iter(["maybe", "y"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    assert ui.ask_approval({"plan": "x"}) == "approve"
    assert "answer" in stream.getvalue()


def test_ask_approval_skips_the_panel_when_the_caller_already_showed_it(monkeypatch) -> None:
    """The in-turn caller's Tasks panel stays above the prompt; only the prompt is owed."""
    stream = _capture(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda *a: "y")

    assert ui.ask_approval({"plan": "[-]  step-a"}, show_plan=False) == "approve"
    assert "step-a" not in stream.getvalue()


def test_ask_question_accepts_a_numbered_choice(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda *a: "2")

    out = ui.ask_question({"type": "question", "question": "which?", "options": ["a", "b", "c"]})

    assert out == {"answer": "b", "cancelled": False}
    printed = stream.getvalue()
    assert "which?" in printed and "1)" in printed and "3)" in printed


def test_ask_question_accepts_free_text(monkeypatch) -> None:
    _capture(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda *a: "neither, use a queue")

    out = ui.ask_question({"type": "question", "question": "which?", "options": ["a", "b"]})

    assert out == {"answer": "neither, use a queue", "cancelled": False}


def test_ask_question_reasks_on_empty_input(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    answers = iter(["", "1"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    out = ui.ask_question({"type": "question", "question": "which?", "options": ["a"]})

    assert out == {"answer": "a", "cancelled": False}
    assert "answer" in stream.getvalue()


def test_ask_question_lets_a_dismissal_escape(monkeypatch) -> None:
    """Policy belongs to the caller: this module only reports that the user walked away."""
    _capture(monkeypatch)

    def walk_away(*_args):
        raise EOFError

    monkeypatch.setattr("builtins.input", walk_away)

    with pytest.raises(EOFError):
        ui.ask_question({"type": "question", "question": "which?", "options": ["a"]})


def test_tool_log_survives_brackets_in_a_path(monkeypatch) -> None:
    """A path like `foo[bar].py` is markup to rich: unescaped it would raise MarkupError."""
    stream = _capture(monkeypatch)
    log = ui.ToolLog()

    log("read_file", {"path": "src/foo[bar].py"}, "l1\nl2")
    log("read_file", {"path": "x"}, "Error: [red]boom[/red]")

    out = stream.getvalue()
    assert "foo[bar].py" in out
    assert "[red]boom[/red]" in out


def test_ask_question_handles_a_unicode_digit(monkeypatch) -> None:
    """`'²'.isdigit()` is True but `int('²')` raises: the prompt must not crash on it."""
    _capture(monkeypatch)
    answers = iter(["²"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    out = ui.ask_question({"type": "question", "question": "which?", "options": ["a", "b"]})

    assert out == {"answer": "²", "cancelled": False}


def test_ask_question_escapes_a_bracketed_option(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    answers = iter(["1"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    out = ui.ask_question(
        {"type": "question", "question": "which[?]", "options": ["use [bold]a[/bold]"]}
    )

    assert out == {"answer": "use [bold]a[/bold]", "cancelled": False}
    assert "which[?]" in stream.getvalue()
