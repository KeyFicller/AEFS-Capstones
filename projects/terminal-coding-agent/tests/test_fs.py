from pathlib import Path

from terminal_coding_agent.tools.fs import build_edit_file, build_read_file

def test_read_file(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    tool = build_read_file(tmp_path)
    assert tool.invoke({"path": "f.txt"}) == "hello"

def test_read_missing(tmp_path: Path) -> None:
    tool = build_read_file(tmp_path)
    out = tool.invoke({"path": "nope.txt"})
    assert out.startswith("Error:")

def test_edit_unique(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a x a\n", encoding="utf-8")
    tool = build_edit_file(tmp_path)
    out = tool.invoke(
        {"path": "f.txt",
        "old_str": "x",
        "new_str": "y"}
    )

    assert "y" in (tmp_path / "f.txt").read_text(encoding="utf-8")
    assert "---" in out or "@@" in out or "+y" in out

def test_edit_zero_matches_no_write(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("abc", encoding="utf-8")
    tool = build_edit_file(tmp_path)
    out = tool.invoke(
        {"path": "f.txt",
        "old_str": "zzz",
        "new_str": "yyy"}
    )

    assert out.startswith("Error:")
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "abc"

def test_edit_multi_matches_no_write(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x x", encoding="utf-8")
    tool = build_edit_file(tmp_path)
    out = tool.invoke({"path": "f.txt", "old_str": "x", "new_str": "y"})
    assert out.startswith("Error:")
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "x x"


def test_edit_escape_is_not_file_not_found(tmp_path: Path) -> None:
    tool = build_edit_file(tmp_path)
    out = tool.invoke({"path": "../outside.txt", "old_str": "a", "new_str": "b"})
    assert out.startswith("Error:")
    assert "escapes" in out.lower()
    assert "file not found" not in out.lower()