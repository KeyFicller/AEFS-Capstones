from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import tool

from terminal_coding_agent.config import SHELL_TIMEOUT_SECONDS
from terminal_coding_agent.tools.truncate import truncate

_ALLOWED = frozenset(
    {
        "init",
        "config",
        "status",
        "diff",
        "log",
        "show",
        "add",
        "commit",
        "checkout",
        "restore",
        "branch",
        "rev-parse",
        "ls-files",
    }
)
_DENIED = frozenset({"push", "pull", "fetch", "clone", "reset"})


def build_git(worktree: Path):
    @tool
    def git(git_args: list[str]) -> str:
        """Run an allowlisted git command in the worktree.

        Args:
            git_args: Git argv after `git` (e.g. ["status"], ["diff", "--stat"]).

        Returns:
            Command output, truncated to ~4000 tokens.
        """
        if not git_args:
            return truncate("Error: git_args is empty")

        subcommand = git_args[0]
        if subcommand in _DENIED or subcommand not in _ALLOWED:
            return truncate(f"Error: git subcommand not allowed: {subcommand}")

        try:
            result = subprocess.run(
                ["git", "-C", str(worktree), *git_args],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=SHELL_TIMEOUT_SECONDS,
            )
        except FileNotFoundError:
            return truncate("Error: git not found")
        except subprocess.TimeoutExpired:
            return truncate(f"Error: git command Timeout: {git_args}")

        return truncate(
            f"exit_code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return git
