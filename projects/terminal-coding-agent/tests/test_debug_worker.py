"""Integration test for the lldb worker: real lldb, real binary. Skipped without a toolchain."""

import json
import os
import select
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

    def send(self, payload: dict, *, timeout: float = 30.0) -> dict:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write((json.dumps(payload) + "\n").encode())
        self.proc.stdin.flush()
        # Without this a blocked worker hangs the whole test session instead of failing.
        if not select.select([self.proc.stdout], [], [], timeout)[0]:
            raise TimeoutError(f"worker did not answer within {timeout}s")
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


def test_a_second_start_on_the_same_target_keeps_the_session(tmp_path: Path) -> None:
    (tmp_path / "bug.cc").write_text(SOURCE)
    subprocess.run(["clang++", "-g", "-O0", "-o", "bug", "bug.cc"], cwd=tmp_path, check=True)
    worker = RawWorker(tmp_path)
    try:
        worker.send({"op": "start", "binary": str(tmp_path / "bug"), "args": []})
        worker.send({"op": "cmd", "command": "breakpoint set -f bug.cc -l 3"})

        again = worker.send({"op": "start", "binary": str(tmp_path / "bug"), "args": []})

        assert "session kept" in again["output"]
        # The breakpoint survives, so the second start did not rebuild the target.
        assert "bug.cc:3" in worker.send({"op": "cmd", "command": "breakpoint list"})["output"]
    finally:
        worker.kill()


def test_a_second_run_in_a_kept_session_does_not_hang(tmp_path: Path) -> None:
    (tmp_path / "bug.cc").write_text(SOURCE)
    subprocess.run(["clang++", "-g", "-O0", "-o", "bug", "bug.cc"], cwd=tmp_path, check=True)
    worker = RawWorker(tmp_path)
    try:
        worker.send({"op": "start", "binary": str(tmp_path / "bug"), "args": []})
        worker.send({"op": "cmd", "command": "breakpoint set -f bug.cc -l 2"})
        first = worker.send({"op": "cmd", "command": "run", "timeout": 10})
        assert first["stop_reason"] == "breakpoint"

        # lldb asks "kill it and restart?" here and reads the answer from stdin; unanswered
        # it blocks the worker forever, which is what killed the session in the smoke run.
        second = worker.send({"op": "cmd", "command": "run", "timeout": 10}, timeout=30)

        assert second["ok"] is True
        assert second["stop_reason"] == "breakpoint"
    finally:
        worker.kill()


def test_changing_the_run_args_rebuilds_the_session(tmp_path: Path) -> None:
    (tmp_path / "bug.cc").write_text(SOURCE)
    subprocess.run(["clang++", "-g", "-O0", "-o", "bug", "bug.cc"], cwd=tmp_path, check=True)
    worker = RawWorker(tmp_path)
    try:
        worker.send({"op": "start", "binary": str(tmp_path / "bug"), "args": []})
        worker.send({"op": "cmd", "command": "breakpoint set -f bug.cc -l 3"})

        again = worker.send({"op": "start", "binary": str(tmp_path / "bug"), "args": ["x"]})

        assert "session kept" not in again["output"]
        # A real rebuild: the old breakpoint is gone.
        assert "bug.cc:3" not in worker.send({"op": "cmd", "command": "breakpoint list"})["output"]
    finally:
        worker.kill()


def test_a_new_stop_reports_its_location(tmp_path: Path) -> None:
    (tmp_path / "bug.cc").write_text(SOURCE)
    subprocess.run(["clang++", "-g", "-O0", "-o", "bug", "bug.cc"], cwd=tmp_path, check=True)
    body_line = next(
        number for number, text in enumerate(SOURCE.splitlines(), start=1) if "int s = 0" in text
    )
    worker = RawWorker(tmp_path)
    try:
        worker.send({"op": "start", "binary": str(tmp_path / "bug"), "args": []})
        worker.send({"op": "cmd", "command": f"breakpoint set -f bug.cc -l {body_line}"})
        # A breakpoint only arms the debugger; nothing has stopped yet.
        assert worker.send({"op": "cmd", "command": "breakpoint list"})["stop_location"] is None

        stopped = worker.send({"op": "cmd", "command": "run", "timeout": 10})
        assert stopped["stop_location"] == {"file": "bug.cc", "line": body_line}
        # Inspecting the frame does not move the stop point: no location, or every
        # `frame variable` would re-print the source line.
        assert worker.send({"op": "cmd", "command": "frame variable"})["stop_location"] is None
        # The same stop point again (a second run) is not news either.
        assert worker.send({"op": "cmd", "command": "run", "timeout": 10})["stop_location"] is None
    finally:
        worker.kill()
