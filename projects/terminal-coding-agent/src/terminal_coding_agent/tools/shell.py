from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import tool

from terminal_coding_agent.config import SHELL_TIMEOUT_SECONDS
from terminal_coding_agent.tools.truncate import truncate


def build_run_shell(worktree: Path):
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
            result = subprocess.run(
                command,
                shell=True,
                cwd=worktree,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return truncate(f"Error: command Timeout: {command}")

        return truncate(
            f"exit_code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return run_shell
