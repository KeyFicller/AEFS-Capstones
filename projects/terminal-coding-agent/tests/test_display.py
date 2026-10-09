"""The tool-side display decorator and the renderer it feeds."""

import io
from typing import Any

from langchain_core.tools import tool
from rich.console import Console
from terminal_coding_agent import ui
from terminal_coding_agent.tools import display
from terminal_coding_agent.tools.fs import build_edit_file
from terminal_coding_agent.tools.search import build_ripgrep, build_tree_sitter_symbols
from terminal_coding_agent.tools.shell import build_run_shell


def _capture(monkeypatch) -> io.StringIO:
    stream = io.StringIO()
    monkeypatch.setattr(
        ui, "CONSOLE", Console(file=stream, width=100, no_color=True, force_terminal=False)
    )
    return stream


def _probe(hook=None):
    """A decorated tool. The decorator MUST sit under @tool, or the schema is lost."""

    @tool
    @display.tool_output(hook)
    def probe(pattern: str, path: str = ".") -> str:
        """Probe.

        Args:
            pattern: the pattern.
            path: where to look.
        """
        return f"matched {pattern}"

    return probe


def test_display_is_off_by_default(monkeypatch) -> None:
    stream = _capture(monkeypatch)

    assert display.LEVEL == "off", "Harbor must not gain console output"
    assert _probe().invoke({"pattern": "TODO"}) == "matched TODO"
    assert stream.getvalue() == ""


def test_header_carries_the_tool_name_and_the_first_argument(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    monkeypatch.setattr(display, "LEVEL", "detail")

    _probe(lambda params, result: ("", f"```text\n{result}\n```")).invoke(
        {"pattern": "TODO", "path": "src"}
    )

    out = stream.getvalue()
    assert "probe" in out and "TODO" in out
    assert "src" not in out, "only the first argument identifies the call"
    assert "matched TODO" in out


def test_simple_prints_only_the_header_line(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    monkeypatch.setattr(display, "LEVEL", "simple")

    _probe(lambda params, result: ("2 hits", "```text\nmatched TODO\n```")).invoke(
        {"pattern": "TODO"}
    )

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1, f"simple is one line per call:\n{stream.getvalue()}"
    assert "probe" in lines[0] and "TODO" in lines[0] and "(2 hits)" in lines[0]
    assert "matched" not in lines[0], "the body is detail-only"


def test_simple_falls_back_to_the_bare_header(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    monkeypatch.setattr(display, "LEVEL", "simple")

    _probe(lambda params, result: ("", "```text\nmatched TODO\n```")).invoke({"pattern": "TODO"})

    line = next(line for line in stream.getvalue().splitlines() if line.strip())
    assert line.strip().endswith("TODO"), "no summary means no parentheses"


def test_detail_prints_the_summary_in_the_header_and_the_body(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    monkeypatch.setattr(display, "LEVEL", "detail")

    _probe(lambda params, result: ("2 hits", "```text\nmatched TODO\n```")).invoke(
        {"pattern": "TODO"}
    )

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert "probe" in lines[0] and "(2 hits)" in lines[0], "the header stays one line"
    assert "matched TODO" in stream.getvalue(), "and detail still prints the body"


def test_target_joins_a_list_argument_and_clips_a_long_one(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    monkeypatch.setattr(display, "LEVEL", "detail")

    @tool
    @display.tool_output()
    def git(git_args: list[str]) -> str:
        """Git.

        Args:
            git_args: argv after git.
        """
        return "exit_code: 0\nstdout:\n\nstderr:\n"

    git.invoke({"git_args": ["status", "--short"]})
    git.invoke({"git_args": ["log", "x" * 200]})

    out = stream.getvalue()
    assert "status --short" in out, "a list argument is joined with spaces"
    assert "x" * 200 not in out and "…" in out


def test_a_no_argument_tool_prints_only_the_name(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    monkeypatch.setattr(display, "LEVEL", "detail")

    @tool
    @display.tool_output()
    def ping() -> str:
        """Ping."""
        return "pong"

    ping.invoke({})

    line = next(line for line in stream.getvalue().splitlines() if "ping" in line)
    assert line.strip().endswith("ping")


def test_an_error_result_is_rendered_without_calling_the_hook(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    monkeypatch.setattr(display, "LEVEL", "detail")
    seen: list[Any] = []

    @tool
    @display.tool_output(lambda params, result: seen.append(result) or ("x", "y"))
    def boom(pattern: str) -> str:
        """Boom.

        Args:
            pattern: the pattern.
        """
        return "Error: old_str not found in a.py"

    boom.invoke({"pattern": "x"})

    assert seen == [], "the hook must not see a failed call"
    assert "old_str not found" in stream.getvalue()


def test_a_raising_hook_does_not_break_the_tool(monkeypatch) -> None:
    stream = _capture(monkeypatch)
    monkeypatch.setattr(display, "LEVEL", "detail")

    def explode(params, result):
        raise ValueError("boom")

    @tool
    @display.tool_output(explode)
    def probe(pattern: str) -> str:
        """Probe.

        Args:
            pattern: the pattern.
        """
        return "ok"

    assert probe.invoke({"pattern": "x"}) == "ok"
    out = stream.getvalue()
    assert "probe" in out
    assert "Traceback" not in out and "boom" not in out


def test_the_decorator_keeps_the_tool_contract() -> None:
    probe = _probe()
    assert probe.name == "probe"
    assert list(probe.args_schema.model_fields) == ["pattern", "path"]


def test_exit_code_and_output_shapes_a_shell_result() -> None:
    result = "exit_code: 3\nstdout:\na\nb\nstderr:\nwarn\n"

    summary, body = display.exit_code_and_output(result, line_count=True)

    assert summary == "exit 3 · 2 lines"
    assert "```text\na\nb\n```" in body
    assert "```text\nwarn\n```" in body, "stderr gets its own block"
    assert display.exit_code_and_output("exit_code: 0\nstdout:\n\nstderr:\n")[0] == "exit 0"
    one_line = display.exit_code_and_output("exit_code: 0\nstdout:\nhi\nstderr:\n", line_count=True)
    assert one_line[0] == "exit 0 · 1 line", "a single line is not '1 lines'"


def _enabled(monkeypatch, level: str = "detail") -> io.StringIO:
    stream = _capture(monkeypatch)
    monkeypatch.setattr(display, "LEVEL", level)
    return stream


def test_edit_file_hook_shows_the_diff(tmp_path, monkeypatch) -> None:
    stream = _enabled(monkeypatch)
    (tmp_path / "a.py").write_text("old\n")

    build_edit_file(tmp_path).invoke({"path": "a.py", "old_str": "old", "new_str": "new"})

    out = stream.getvalue()
    assert "+1 -1" in out, "a compact summary leads the body"
    assert "-old" in out and "+new" in out, "the diff itself must be shown"


def test_run_shell_hook_shows_the_exit_code_and_output(tmp_path, monkeypatch) -> None:
    tool = build_run_shell(tmp_path)

    detail = _enabled(monkeypatch, "detail")
    tool.invoke({"command": "echo hi"})
    out = detail.getvalue()
    assert "(exit 0 · 1 line)" in out.splitlines()[0], "exact, so the plural form is caught too"
    assert "hi" in out

    simple = _enabled(monkeypatch, "simple")
    tool.invoke({"command": "echo hi"})
    assert "(exit 0 · 1 line)" in simple.getvalue(), "the simple header looks the same"


def test_ripgrep_hook_counts_hits(tmp_path, monkeypatch) -> None:
    stream = _enabled(monkeypatch, "simple")
    (tmp_path / "a.py").write_text("TODO one\nTODO two\n")

    build_ripgrep(tmp_path).invoke({"pattern": "TODO"})

    assert "(2 hits)" in stream.getvalue()


def test_read_file_and_symbols_report_a_summary(tmp_path, monkeypatch) -> None:
    stream = _enabled(monkeypatch, "simple")
    (tmp_path / "a.py").write_text("def one():\n    pass\n\n\ndef two():\n    pass\n")

    from terminal_coding_agent.tools.fs import build_read_file

    build_read_file(tmp_path).invoke({"path": "a.py"})
    build_tree_sitter_symbols(tmp_path).invoke({"path": "a.py"})

    out = stream.getvalue()
    assert "(6 lines)" in out, "the simple level is useless without a summary"
    assert "(2 symbols)" in out


def test_a_failed_tool_call_skips_its_hook(tmp_path, monkeypatch) -> None:
    stream = _enabled(monkeypatch)

    build_edit_file(tmp_path).invoke({"path": "a.py", "old_str": "old", "new_str": "new"})

    out = stream.getvalue()
    assert "file not found" in out, "the error is surfaced"
    assert "```diff" not in out, "and the hook never ran"
