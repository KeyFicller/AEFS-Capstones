"""Publish the worktree diff so a finished run can be inspected after the fact.

Harbor mounts ``/logs/artifacts`` into the trial container and collects it by
convention, so a patch written there outlives the trial (and is the only reason
a run's edits are visible at all); a local run keeps a copy under the worktree.
"""

import logging
import subprocess
from pathlib import Path

from terminal_coding_agent.config import SHELL_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

PATCH_FILENAME = "patch.diff"
PATCH_RELPATH = f".agent/{PATCH_FILENAME}"
HARBOR_ARTIFACTS_DIR = Path("/logs/artifacts")


def _git(worktree: Path, *args: str) -> str:
    """Read-only git query in the worktree; "" when git cannot answer."""
    try:
        result = subprocess.run(
            ["git", "-C", str(worktree), *args],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=SHELL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def worktree_patch(worktree: Path) -> str:
    """Everything uncommitted since HEAD, staged edits plus untracked paths.

    ``git diff HEAD`` (not a bare ``git diff``) also shows staged edits, so a run
    that staged its fix still appears here. A run that *committed* its work is out
    of scope: HEAD moves with it, so the diff is legitimately empty -- the header
    says so rather than reporting a silent "no changes".
    """
    status = _git(worktree, "status", "--porcelain")
    diff = _git(worktree, "diff", "HEAD")
    return (
        f"# git status --porcelain\n{status}\n"
        f"# git diff HEAD (uncommitted changes; a committed fix does not appear)\n{diff}"
    )


def publish_patch(worktree: Path) -> list[Path]:
    """Write :func:`worktree_patch` to the worktree and, when mounted, to Harbor.

    Never raises: it runs from the graph's Stop hook, where an exception would
    turn an otherwise finished run into a trial failure.
    """
    patch = worktree_patch(worktree)
    targets = [worktree / PATCH_RELPATH]
    if HARBOR_ARTIFACTS_DIR.is_dir():
        targets.append(HARBOR_ARTIFACTS_DIR / PATCH_FILENAME)

    written: list[Path] = []
    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(patch, encoding="utf-8")
        except OSError:
            logger.warning("could not write patch to %s", target, exc_info=True)
            continue
        written.append(target)
    return written
