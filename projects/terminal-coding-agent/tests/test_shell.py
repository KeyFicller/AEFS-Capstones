import os
from pathlib import Path

import pytest
from terminal_coding_agent.config import SHELL_TIMEOUT_SECONDS
from terminal_coding_agent.sandbox import SandboxError
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


class _FakeSandbox:
    def __init__(self, *, result=(0, "from-container\n", ""), error=None) -> None:
        self.calls: list[tuple[str, int]] = []
        self._result = result
        self._error = error

    def run(self, command: str, timeout: int):
        self.calls.append((command, timeout))
        if self._error is not None:
            raise self._error
        return self._result


def test_run_shell_uses_the_sandbox(tmp_path: Path) -> None:
    box = _FakeSandbox()

    out = build_run_shell(tmp_path, sandbox=box).invoke({"command": "echo hi"})

    assert "from-container" in out
    assert box.calls == [("echo hi", SHELL_TIMEOUT_SECONDS)]


def test_run_shell_reports_a_sandbox_failure(tmp_path: Path) -> None:
    box = _FakeSandbox(error=SandboxError("container is gone"))

    out = build_run_shell(tmp_path, sandbox=box).invoke({"command": "echo hi"})

    assert out.startswith("Error: could not run command")
    assert "container is gone" in out


def test_run_shell_reports_a_sandbox_timeout(tmp_path: Path) -> None:
    box = _FakeSandbox(result=None)

    out = build_run_shell(tmp_path, sandbox=box).invoke({"command": "sleep 9", "timeout_sec": 2})

    assert out.startswith("Error: command Timeout after 2s")
