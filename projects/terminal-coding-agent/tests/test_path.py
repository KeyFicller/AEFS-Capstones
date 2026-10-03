from pathlib import Path

from terminal_coding_agent.tools.path import resolve_in_worktree


def test_resolves_relative_inside(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    got = resolve_in_worktree(tmp_path, "a.txt")

    assert got == (tmp_path / "a.txt").resolve()


def test_rejects_dotdot_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"

    got = resolve_in_worktree(tmp_path, "../outside.txt")

    assert isinstance(got, str) and got.startswith("Error:")
    assert not outside.exists()


def test_rejects_absolute_outside(tmp_path: Path) -> None:
    got = resolve_in_worktree(tmp_path, "/etc/passwd")

    assert isinstance(got, str) and got.startswith("Error:")
