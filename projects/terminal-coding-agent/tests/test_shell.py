from pathlib import Path

from terminal_coding_agent.tools.shell import build_run_shell


def test_run_shell_timeout(tmp_path: Path) -> None:
    tool = build_run_shell(tmp_path)
    out = tool.invoke({"command": "sleep 5", "timeout_sec": 1})
    assert "Timeout" in out and out.startswith("Error:")


def test_run_shell_default_timeout_arg_optional(tmp_path: Path) -> None:
    tool = build_run_shell(tmp_path)
    out = tool.invoke({"command": "echo hi"})
    assert "exit_code: 0" in out
    assert "hi" in out


def test_run_shell_survives_non_utf8_output(tmp_path: Path) -> None:
    """Binary output (e.g. cat on a MARC .mrc) must not crash the tool with UnicodeDecodeError."""
    (tmp_path / "blob.bin").write_bytes(b"\x00\x95\xfe\xffMARC")
    tool = build_run_shell(tmp_path)
    out = tool.invoke({"command": "cat blob.bin"})
    assert "exit_code: 0" in out
