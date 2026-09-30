from pathlib import Path

from terminal_coding_agent.demo import DEMO_TASK, seed_demo_worktree


def test_seed_demo_worktree(tmp_path: Path) -> None:
    seed_demo_worktree(tmp_path)
    assert (tmp_path / "greeter.py").is_file()
    assert (tmp_path / ".git").is_dir()
    text = (tmp_path / "greeter.py").read_text(encoding="utf-8")
    assert "Helo," in text
    assert "SystemExit" in text
    assert "Hello, agent" in DEMO_TASK
    assert "python3 greeter.py" in DEMO_TASK
    assert "report_blocked" in DEMO_TASK
