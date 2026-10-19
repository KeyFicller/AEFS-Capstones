"""Seeded worktree + prompt for the local graph.py demo."""

import os
import subprocess
from pathlib import Path

from terminal_coding_agent.config import SHELL_TIMEOUT_SECONDS

DEMO_TASK = """\
You are working in an existing git worktree (already `git init`'d).

`python3 greeter.py` should print exactly `Hello, agent` and exit 0. It currently
raises SystemExit before printing.

1. Run `python3 greeter.py` with `run_shell` and read the failure.
2. Edit `greeter.py` so `greet` returns `Hello, {name}`, and replace the
   SystemExit with `print(greet("agent"))`.
3. Run `python3 greeter.py`. This step is done when stdout is exactly `Hello, agent`
   and the exit code is 0.
4. Commit `greeter.py` only (`git add greeter.py`, then `git commit`). Untracked
   `.agent/` or `__pycache__/` are expected and are not a failure.

Prefer the dedicated tools over reinventing them in the shell. No network.
"""

_GREETER_PY = '''\
"""Tiny greeter used by the coding-agent demo."""


def greet(name: str) -> str:
    return f"Helo, {name}"


if __name__ == "__main__":
    raise SystemExit("greeting is wrong")
'''

_README = """# demo worktree

`python3 greeter.py` exits with SystemExit. It should print `Hello, agent`.
"""


def _git(worktree: Path, *args: str) -> None:
    """Run one git command with a bounded timeout and no interactive prompts."""
    subprocess.run(
        ["git", *args],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
        timeout=SHELL_TIMEOUT_SECONDS,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_EDITOR": "true"},
    )


def seed_demo_worktree(worktree: Path) -> None:
    """Write starter files and create an initial git commit (git init is not tool-allowlisted)."""
    worktree = worktree.resolve()
    worktree.mkdir(parents=True, exist_ok=True)
    (worktree / "greeter.py").write_text(_GREETER_PY, encoding="utf-8")
    (worktree / "README.md").write_text(_README, encoding="utf-8")

    _git(worktree, "init")
    _git(worktree, "config", "user.email", "demo@example.com")
    _git(worktree, "config", "user.name", "Demo Agent")
    _git(worktree, "add", "greeter.py", "README.md")
    # Idempotent: on an already-seeded worktree (same bytes) `git add` stages nothing
    # and a plain `git commit` would fail with "nothing to commit" and check=True.
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
        timeout=SHELL_TIMEOUT_SECONDS,
    ).stdout
    if status.strip():
        _git(worktree, "commit", "-m", "chore: seed broken greeter for demo")
