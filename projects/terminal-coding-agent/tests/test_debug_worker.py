"""Integration test for the lldb worker: real lldb, real binary. Skipped without a toolchain."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import terminal_coding_agent

SYSTEM_PYTHON = Path("/usr/bin/python3")
WORKER = Path(terminal_coding_agent.__file__).parent / "debug_worker.py"
SOURCE = """
struct Node { int v; Node* next; };
int sum(Node* p) { int s = 0; while (p->next) { s += p->v; p = p->next; } return s; }
int main() { Node* p = nullptr; return sum(p); }
"""

pytestmark = pytest.mark.skipif(
    shutil.which("lldb") is None or shutil.which("clang++") is None or not SYSTEM_PYTHON.exists(),
    reason="needs lldb bindings, clang++ and the system python3",
)


def _env() -> dict:
    lldb = shutil.which("lldb")
    python_path = subprocess.run(
        [lldb, "-P"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {**os.environ, "PYTHONPATH": python_path}


class RawWorker:
    def __init__(self, worktree: Path) -> None:
        self.proc = subprocess.Popen(
            [str(SYSTEM_PYTHON), str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=worktree,
            env=_env(),
            start_new_session=True,
        )

    def send(self, payload: dict) -> dict:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write((json.dumps(payload) + "\n").encode())
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def kill(self) -> None:
        self.proc.kill()


def test_start_run_and_inspect(tmp_path: Path) -> None:
    (tmp_path / "bug.cc").write_text(SOURCE)
    subprocess.run(["clang++", "-g", "-O0", "-o", "bug", "bug.cc"], cwd=tmp_path, check=True)
    worker = RawWorker(tmp_path)
    try:
        started = worker.send({"op": "start", "binary": str(tmp_path / "bug"), "args": []})
        assert started["ok"] is True

        assert worker.send({"op": "cmd", "command": "breakpoint set -f bug.cc -l 3"})["ok"] is True

        stopped = worker.send({"op": "cmd", "command": "run", "timeout": 30})
        assert stopped["stop_reason"] == "breakpoint"

        crash = worker.send({"op": "cmd", "command": "continue", "timeout": 30})
        assert crash["stop_reason"] == "exception"
        assert "nullptr" in worker.send({"op": "cmd", "command": "frame variable p"})["output"]
    finally:
        worker.kill()


def test_command_without_a_session_is_reported() -> None:
    worker = RawWorker(Path.cwd())
    try:
        response = worker.send({"op": "cmd", "command": "run"})
        assert response["ok"] is False
        assert "debug_start" in response["output"]
    finally:
        worker.kill()


def test_unknown_op_does_not_kill_the_worker(tmp_path: Path) -> None:
    worker = RawWorker(tmp_path)
    try:
        assert worker.send({"op": "bogus"})["ok"] is False
        # still answering a second request (no session yet, but the worker is alive)
        assert "no session" in worker.send({"op": "cmd", "command": "help"})["output"]
    finally:
        worker.kill()
