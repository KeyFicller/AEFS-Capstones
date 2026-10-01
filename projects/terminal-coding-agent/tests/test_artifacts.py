from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from terminal_coding_agent import artifacts
from terminal_coding_agent.artifacts import PATCH_RELPATH, publish_patch, worktree_patch


def _repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "app.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=path, check=True)
    return path


def test_worktree_patch_reports_edits_and_untracked_files(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    (repo / "app.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "repro.py").write_text("print('repro')\n", encoding="utf-8")

    patch_text = worktree_patch(repo)

    assert "-x = 1" in patch_text
    assert "+x = 2" in patch_text
    # A diff cannot show a new file; only the status section can.
    assert "repro.py" in patch_text


def test_worktree_patch_includes_staged_edits(tmp_path: Path) -> None:
    """A bare `git diff` would miss this; the patch prefers `git diff HEAD`."""
    repo = _repo(tmp_path / "repo")
    (repo / "app.py").write_text("x = 3\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=repo, check=True)

    assert "+x = 3" in worktree_patch(repo)


def test_publish_patch_writes_the_worktree_copy(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    (repo / "app.py").write_text("x = 2\n", encoding="utf-8")

    with patch.object(artifacts, "HARBOR_ARTIFACTS_DIR", tmp_path / "absent"):
        written = publish_patch(repo)

    assert written == [repo / PATCH_RELPATH]
    assert "-x = 1" in (repo / PATCH_RELPATH).read_text(encoding="utf-8")


def test_publish_patch_also_targets_the_mounted_harbor_dir(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    (repo / "app.py").write_text("x = 2\n", encoding="utf-8")
    harbor_dir = tmp_path / "logs" / "artifacts"
    harbor_dir.mkdir(parents=True)

    with patch.object(artifacts, "HARBOR_ARTIFACTS_DIR", harbor_dir):
        written = publish_patch(repo)

    assert harbor_dir / "patch.diff" in written
    assert (harbor_dir / "patch.diff").read_text(
        encoding="utf-8"
    ) == (repo / PATCH_RELPATH).read_text(encoding="utf-8")


def test_publish_patch_never_raises_when_git_is_missing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"

    with (
        patch.object(artifacts.subprocess, "run", side_effect=FileNotFoundError("git")),
        patch.object(artifacts, "HARBOR_ARTIFACTS_DIR", tmp_path / "absent"),
    ):
        written = publish_patch(repo)

    assert written == [repo / PATCH_RELPATH]
