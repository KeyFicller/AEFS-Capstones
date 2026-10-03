import shutil
from pathlib import Path

import pytest
from terminal_coding_agent.tools.search import build_ripgrep, build_tree_sitter_symbols


@pytest.mark.skipif(shutil.which("rg") is None, reason="rg not installed")
def test_ripgrep_finds_line(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("foo = 1\n", encoding="utf-8")
    tool = build_ripgrep(tmp_path)
    out = tool.invoke({"pattern": "foo", "path": "."})
    assert "foo" in out
    assert not out.startswith("Error")


def test_symbols_python(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text("def f():\n    pass\n\nclass C:\n    pass\n", encoding="utf-8")
    tool = build_tree_sitter_symbols(tmp_path)
    out = tool.invoke({"path": "m.py"})
    assert "f" in out and "C" in out


def test_symbols_non_python(tmp_path: Path) -> None:
    (tmp_path / "a.js").write_text("function f() {}", encoding="utf-8")
    tool = build_tree_sitter_symbols(tmp_path)
    out = tool.invoke({"path": "a.js"})
    assert out.startswith("Error:")
