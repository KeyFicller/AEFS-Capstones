"""lldb debug tools backed by a persistent worker subprocess.

The lldb Python bindings are built for Python 3.9 (``_lldb.cpython-39-*``) while
the package runs on 3.14, so the live session lives in a worker we launch with the
system interpreter and drive over newline-delimited JSON. Nothing here parses
console text: the worker calls ``SBDebugger.HandleCommand`` and returns structured
results.
"""

import atexit
import json
import os
import select
import shutil
import signal
import subprocess
import time
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from terminal_coding_agent.config import DEBUG_START_TIMEOUT_SECONDS, DEBUG_TIMEOUT_SECONDS
from terminal_coding_agent.tools.display import tool_output
from terminal_coding_agent.tools.path import resolve_in_worktree
from terminal_coding_agent.tools.truncate import truncate

_WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "debug_worker.py"
_SYSTEM_PYTHON = Path("/usr/bin/python3")
_READ_MARGIN_SECONDS = 15


def lldb_python_path() -> str | None:
    """Directory holding lldb's Python bindings (``lldb -P``), or None when lldb is absent."""
    lldb = shutil.which("lldb")
    if lldb is None:
        return None
    try:
        result = subprocess.run(
            [lldb, "-P"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


class Worker:
    """The lldb worker process and its JSON-line framing."""

    def __init__(self, worktree: Path) -> None:
        self._worktree = worktree
        self._proc: subprocess.Popen | None = None
        self._buffer = b""

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        python_path = lldb_python_path()
        if python_path is None:
            raise RuntimeError("lldb not found; install the Xcode Command Line Tools")
        if not _SYSTEM_PYTHON.exists():
            raise RuntimeError(f"{_SYSTEM_PYTHON} not found; debug needs the system python3")
        self._proc = subprocess.Popen(
            [str(_SYSTEM_PYTHON), str(_WORKER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=self._worktree,
            env={**os.environ, "PYTHONPATH": python_path},
            start_new_session=True,
        )
        atexit.register(self.kill)

    def request(self, payload: dict, *, timeout: float) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("debug session is not running")
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write((json.dumps(payload) + "\n").encode())
        self._proc.stdin.flush()
        deadline = time.time() + timeout
        while b"\n" not in self._buffer:
            remaining = deadline - time.time()
            if remaining <= 0:
                self.kill()
                raise TimeoutError("debug worker did not respond")
            ready, _, _ = select.select([self._proc.stdout], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(self._proc.stdout.fileno(), 65536)
            if not chunk:
                self.kill()
                raise RuntimeError("debug worker exited")
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return json.loads(line.decode("utf-8"))

    def kill(self) -> None:
        proc = self._proc
        self._proc = None
        self._buffer = b""
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait()


def _render(response: dict) -> str:
    text = ("" if response.get("ok") else "Error: ") + str(response.get("output") or "")
    reason = response.get("stop_reason")
    if reason:
        text += f"\nstop_reason: {reason}"
    return truncate(text)


def _source_window(worktree: Path, location: dict, *, radius: int = 1) -> str:
    """The stopped-at line plus `radius` around it, `>` marking the current one."""
    resolved = resolve_in_worktree(worktree, location["file"])
    if isinstance(resolved, str):
        return ""
    try:
        source = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    current = location["line"]
    first, last = max(1, current - radius), min(len(source), current + radius)
    return "\n".join(
        f"{'>' if number == current else ' '} {number:>3}  {source[number - 1]}"
        for number in range(first, last + 1)
    )


def build_debug(worktree: Path) -> list[BaseTool]:
    worker = Worker(worktree)
    # The raw response of the last command, for the display hook: the model-facing
    # string has already rendered `stop_location` away, and one worker runs one
    # command at a time, so a single slot is enough.
    last: dict = {}

    def send(payload: dict, *, timeout: float) -> str:
        try:
            worker.start()
            response = worker.request(payload, timeout=timeout)
        except (RuntimeError, TimeoutError, OSError) as exc:
            last.clear()
            return f"Error: {exc}"
        last.clear()
        last.update(response)
        return _render(response)

    def _display_cmd(params: dict, result: str) -> tuple[str, str]:
        summary = ""
        blocks: list[str] = []
        reason = last.get("stop_reason")
        location = last.get("stop_location")
        if reason:
            # `_render` repeats the stop reason for the model; in the log it is either
            # duplicated by the summary below or stale from an earlier stop.
            result = result.replace(f"\nstop_reason: {reason}", "")
        if location:
            summary = f"stop {reason} · {location['file']}:{location['line']}"
            window = _source_window(worktree, location)
            if window:
                blocks.append(f"```c\n{window}\n```")
        blocks.append(f"```text\n{result}\n```")
        return summary, "\n\n".join(blocks)

    @tool
    def debug_start(binary: str, run_args: list[str] | None = None) -> str:
        """Start or restart a debug session on a binary built with `-g`.

        Args:
            binary: Executable path, relative to the worktree.
            run_args: Optional command-line arguments for the program.

        Returns:
            Confirmation, or an Error line when the target cannot be created.
        """
        resolved = resolve_in_worktree(worktree, binary)
        if isinstance(resolved, str):
            return resolved
        return send(
            {"op": "start", "binary": str(resolved), "args": list(run_args or [])},
            timeout=DEBUG_START_TIMEOUT_SECONDS,
        )

    @tool
    @tool_output(_display_cmd)
    def debug_cmd(command: str, timeout_sec: int | None = None) -> str:
        """Run one lldb command in the live session.

        Args:
            command: An lldb command line, e.g. `breakpoint set -f main.cpp -l 42`,
                `run`, `frame variable`, `thread backtrace`, `continue`.
            timeout_sec: Wall-clock limit for this command; defaults to the
                configured debug timeout. `run`/`continue` are interrupted when
                they exceed it, leaving the session stopped and reusable.

        Returns:
            The command's output, truncated, plus its stop reason when it stopped.
        """
        timeout = timeout_sec if timeout_sec is not None else DEBUG_TIMEOUT_SECONDS
        return send(
            {"op": "cmd", "command": command, "timeout": timeout},
            timeout=timeout + _READ_MARGIN_SECONDS,
        )

    @tool
    def debug_stop() -> str:
        """End the debug session and release the worker process."""
        send({"op": "stop"}, timeout=DEBUG_TIMEOUT_SECONDS)
        worker.kill()
        return "debug session ended"

    return [debug_start, debug_cmd, debug_stop]
