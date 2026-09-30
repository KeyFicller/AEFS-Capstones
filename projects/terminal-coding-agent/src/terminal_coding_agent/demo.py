"""Seeded worktree + prompt for the local graph.py demo."""

from __future__ import annotations

import subprocess
from pathlib import Path

DEMO_TASK = """\
You are working in an existing git worktree (already `git init`'d).

There is a bug in the greeting printed by `greeter.py`: it should print exactly
`Hello, agent` (with that capitalization and comma), but currently it does not.

Do the following using your tools (stay inside the worktree; no network):
1. Use `tree_sitter_symbols` and/or `ripgrep` to find where the greeting string is built.
2. `read_file` the relevant file(s), then `edit_file` to fix the bug (unique old_str → new_str).
3. `run_shell` to run `python greeter.py` and confirm the output is exactly `Hello, agent`.
4. Use `git` (`status` / `diff` / `add` / `commit`) to record the fix with a short message.

Prefer the dedicated tools over reinventing them in the shell.
"""

_GREETER_PY = '''\
"""Tiny greeter used by the coding-agent demo."""


def greet(name: str) -> str:
    # Intentional typo for the demo agent to find and fix.
    return f"Helo, {name}"


if __name__ == "__main__":
    print(greet("agent"))
'''

_README = """# demo worktree

Broken on purpose: `python greeter.py` should print `Hello, agent`.
"""


def seed_demo_worktree(worktree: Path) -> None:
    """Write starter files and create an initial git commit (git init is not tool-allowlisted)."""
    worktree = worktree.resolve()
    worktree.mkdir(parents=True, exist_ok=True)
    (worktree / "greeter.py").write_text(_GREETER_PY, encoding="utf-8")
    (worktree / "README.md").write_text(_README, encoding="utf-8")

    subprocess.run(["git", "init"], cwd=worktree, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "demo@example.com"], cwd=worktree, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Demo Agent"], cwd=worktree, check=True, capture_output=True)
    subprocess.run(["git", "add", "greeter.py", "README.md"], cwd=worktree, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "chore: seed broken greeter for demo"],
        cwd=worktree,
        check=True,
        capture_output=True,
    )
