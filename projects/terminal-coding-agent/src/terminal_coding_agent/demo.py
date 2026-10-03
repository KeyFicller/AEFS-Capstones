"""Seeded worktree + prompt for the local graph.py demo."""

import subprocess
from pathlib import Path

DEMO_TASK = """\
You are working in an existing git worktree (already `git init`'d).

`python3 greeter.py` should print exactly `Hello, agent` and exit 0. It currently
raises SystemExit before printing. Keep the steps below separate. Do not merge them.

1. Verification only. Run `python3 greeter.py` with `run_shell`. Do not read or
   edit any file in this step. The command will exit non-zero. Call `report_blocked`
   with that command and its error. Do not retry and do not try to fix it here.
2. Edit `greeter.py` so `greet` returns `Hello, {name}`, and replace the
   SystemExit with `print(greet("agent"))`. The edit is finished when the file
   would print `Hello, agent`. Do not call `report_blocked` if the edit succeeds.
3. Run `python3 greeter.py`. This step is done when stdout is exactly `Hello, agent`
   and the exit code is 0.
4. Commit `greeter.py` only (`git add greeter.py`, then `git commit`). Untracked
   `.agent/` or `__pycache__/` are expected and are not a failure.

`report_blocked` is only for step 1. Steps 2-4 may read, edit, run, and commit.
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


def seed_demo_worktree(worktree: Path) -> None:
    """Write starter files and create an initial git commit (git init is not tool-allowlisted)."""
    worktree = worktree.resolve()
    worktree.mkdir(parents=True, exist_ok=True)
    (worktree / "greeter.py").write_text(_GREETER_PY, encoding="utf-8")
    (worktree / "README.md").write_text(_README, encoding="utf-8")

    subprocess.run(["git", "init"], cwd=worktree, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "demo@example.com"],
        cwd=worktree,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Demo Agent"], cwd=worktree, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "add", "greeter.py", "README.md"], cwd=worktree, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "chore: seed broken greeter for demo"],
        cwd=worktree,
        check=True,
        capture_output=True,
    )
