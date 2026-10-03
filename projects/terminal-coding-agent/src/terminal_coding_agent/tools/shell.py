import os
import signal
import subprocess
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from terminal_coding_agent.config import SHELL_TIMEOUT_SECONDS
from terminal_coding_agent.tools.truncate import truncate


def _run(command: str, worktree: Path, timeout: int) -> tuple[int, str, str] | None:
    """Run `command` in its own process group; None when it had to be killed.

    A new session (process group) is what makes the timeout real: `subprocess.run`
    kills only the shell wrapper, so `server &` or `nohup … &` keep running as
    orphans after the tool returns. `stdin` is /dev/null so a command that reads
    (git credential prompt, an editor, `cat`) cannot steal the agent's terminal.
    """
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=worktree,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        return None
    return process.returncode, stdout, stderr


def _kill_group(process: subprocess.Popen) -> None:
    """SIGKILL the whole session, then reap it so no zombie is left."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    process.communicate()


def build_run_shell(worktree: Path) -> BaseTool:
    @tool
    def run_shell(command: str, timeout_sec: int | None = None) -> str:
        """Run a shell command in the worktree and return the output.

        Args:
            command: The command to run.
            timeout_sec: Optional timeout in seconds (default from config).

        Returns:
            exit_code / stdout / stderr, truncated to ~4000 tokens.
        """
        timeout = timeout_sec if timeout_sec is not None else SHELL_TIMEOUT_SECONDS
        try:
            result = _run(command, worktree, timeout)
        except OSError as exc:
            return truncate(f"Error: could not run command: {exc}")

        if result is None:
            return truncate(f"Error: command Timeout after {timeout}s: {command}")

        exit_code, stdout, stderr = result
        return truncate(f"exit_code: {exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}")

    return run_shell
