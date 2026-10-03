import subprocess
from pathlib import Path

from terminal_coding_agent.tools.git import build_git


def test_git_push_rejected(tmp_path: Path) -> None:
    tool = build_git(tmp_path)
    out = tool.invoke({"git_args": ["push", "origin", "main"]})
    assert out.startswith("Error:")
    assert "not allowed" in out.lower()


def test_git_init_allowed(tmp_path: Path) -> None:
    tool = build_git(tmp_path)
    out = tool.invoke({"git_args": ["init"]})
    assert out.startswith("exit_code:")
    assert "exit_code: 0" in out
    assert (tmp_path / ".git").exists()


def test_git_reset_rejected(tmp_path: Path) -> None:
    tool = build_git(tmp_path)
    out = tool.invoke({"git_args": ["reset", "--hard"]})
    assert out.startswith("Error:")
    assert "not allowed" in out.lower()


def test_git_config_rejected(tmp_path: Path) -> None:
    """`core.editor` / `core.hooksPath` (and --global) turn later git calls into execution."""
    tool = build_git(tmp_path)
    out = tool.invoke({"git_args": ["config", "--global", "core.editor", "sh -c evil"]})
    assert out.startswith("Error:")
    assert "not allowed" in out.lower()


def test_git_status(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    tool = build_git(tmp_path)
    out = tool.invoke({"git_args": ["status"]})
    assert out.startswith("exit_code:")
    assert "exit_code: 0" in out
