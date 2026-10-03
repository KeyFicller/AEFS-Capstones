import os
from pathlib import Path

import pytest
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


def test_run_shell_does_not_read_the_parent_stdin(tmp_path: Path) -> None:
    """An interactive command must not be able to block on (or eat) the agent's terminal."""
    tool = build_run_shell(tmp_path)
    out = tool.invoke({"command": "cat", "timeout_sec": 5})
    assert "exit_code: 0" in out


def test_run_shell_timeout_kills_background_descendants(tmp_path: Path) -> None:
    """`sleep 30 &` must not outlive the tool call as an orphan process."""
    tool = build_run_shell(tmp_path)
    marker = tmp_path / "pid"
    out = tool.invoke(
        {
            "command": f"sh -c 'sleep 30 & echo $! > {marker}' ; sleep 30",
            "timeout_sec": 1,
        }
    )
    assert out.startswith("Error:")

    pid = int(marker.read_text(encoding="utf-8").strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
