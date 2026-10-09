"""Tool-layer tests, using a fake worker so they need no lldb at all."""

import shutil
import subprocess
from pathlib import Path

import pytest
from terminal_coding_agent.tools import debug as debug_tools

SOURCE = """
struct Node { int v; Node* next; };
int sum(Node* p) { int s = 0; while (p->next) { s += p->v; p = p->next; } return s; }
int main() { Node* p = nullptr; return sum(p); }
"""

needs_toolchain = pytest.mark.skipif(
    debug_tools.lldb_python_path() is None
    or shutil.which("clang++") is None
    or not Path("/usr/bin/python3").exists(),
    reason="needs lldb bindings, clang++ and the system python3",
)


class FakeWorker:
    instances: list["FakeWorker"] = []
    response: dict = {"ok": True, "output": "ok", "stop_reason": None}

    def __init__(self, worktree: Path) -> None:
        self.worktree = worktree
        self.requests: list[dict] = []
        self.killed = False
        FakeWorker.instances.append(self)

    def start(self) -> None:
        pass

    def request(self, payload: dict, *, timeout: float) -> dict:
        self.requests.append(payload)
        return FakeWorker.response

    def kill(self) -> None:
        self.killed = True


@pytest.fixture
def fake(monkeypatch):
    FakeWorker.instances = []
    FakeWorker.response = {"ok": True, "output": "ok", "stop_reason": None}
    monkeypatch.setattr(debug_tools, "Worker", FakeWorker)
    return FakeWorker


def _tools(worktree: Path) -> dict:
    return {t.name: t for t in debug_tools.build_debug(worktree)}


def test_start_sends_a_start_request(tmp_path: Path, fake) -> None:
    _tools(tmp_path)["debug_start"].invoke({"binary": "bug"})
    assert fake.instances[0].requests[-1] == {
        "op": "start",
        "binary": str((tmp_path / "bug").resolve()),
        "args": [],
    }


def test_cmd_uses_the_configured_timeout(tmp_path: Path, fake) -> None:
    _tools(tmp_path)["debug_cmd"].invoke({"command": "run"})
    request = fake.instances[0].requests[-1]
    assert request["op"] == "cmd" and request["command"] == "run"
    assert request["timeout"] == debug_tools.DEBUG_TIMEOUT_SECONDS


def test_cmd_honours_an_explicit_timeout(tmp_path: Path, fake) -> None:
    _tools(tmp_path)["debug_cmd"].invoke({"command": "run", "timeout_sec": 5})
    assert fake.instances[0].requests[-1]["timeout"] == 5


def test_failed_command_is_prefixed_with_error(tmp_path: Path, fake) -> None:
    fake.response = {
        "ok": False,
        "output": "no session; call debug_start first",
        "stop_reason": None,
    }
    assert _tools(tmp_path)["debug_cmd"].invoke({"command": "run"}).startswith("Error: ")


def test_stop_reason_is_surfaced(tmp_path: Path, fake) -> None:
    fake.response = {"ok": True, "output": "stopped", "stop_reason": "breakpoint"}
    assert "stop_reason: breakpoint" in _tools(tmp_path)["debug_cmd"].invoke({"command": "run"})


def test_output_is_truncated(tmp_path: Path, fake) -> None:
    fake.response = {"ok": True, "output": "x" * 100_000, "stop_reason": None}
    assert len(_tools(tmp_path)["debug_cmd"].invoke({"command": "run"})) < 100_000


def test_path_escape_is_rejected_before_touching_the_worker(tmp_path: Path, fake) -> None:
    result = _tools(tmp_path)["debug_start"].invoke({"binary": "../outside"})
    assert result.startswith("Error: ")
    assert fake.instances[0].requests == []


def test_stop_ends_the_session(tmp_path: Path, fake) -> None:
    tools = _tools(tmp_path)
    tools["debug_start"].invoke({"binary": "bug"})
    assert "ended" in tools["debug_stop"].invoke({})
    assert fake.instances[0].killed is True


@needs_toolchain
def test_end_to_end_breakpoint_and_crash(tmp_path: Path) -> None:
    (tmp_path / "bug.cc").write_text(SOURCE)
    subprocess.run(["clang++", "-g", "-O0", "-o", "bug", "bug.cc"], cwd=tmp_path, check=True)
    tools = _tools(tmp_path)
    try:
        assert "target:" in tools["debug_start"].invoke({"binary": "bug"})
        assert "Breakpoint" in tools["debug_cmd"].invoke(
            {"command": "breakpoint set -f bug.cc -l 3"}
        )
        assert "stop_reason: breakpoint" in tools["debug_cmd"].invoke({"command": "run"})
        assert "stop_reason: exception" in tools["debug_cmd"].invoke({"command": "continue"})
        assert "nullptr" in tools["debug_cmd"].invoke({"command": "frame variable p"})
    finally:
        tools["debug_stop"].invoke({})


@needs_toolchain
def test_hung_run_is_interrupted_and_the_session_survives(tmp_path: Path) -> None:
    (tmp_path / "spin.cc").write_text(
        "int main() { volatile long x = 0; while (true) { x++; } return 0; }"
    )
    subprocess.run(["clang++", "-g", "-O0", "-o", "spin", "spin.cc"], cwd=tmp_path, check=True)
    tools = _tools(tmp_path)
    try:
        tools["debug_start"].invoke({"binary": "spin"})
        assert "interrupted" in tools["debug_cmd"].invoke({"command": "run", "timeout_sec": 2})
        assert "main" in tools["debug_cmd"].invoke({"command": "thread backtrace"})
    finally:
        tools["debug_stop"].invoke({})
