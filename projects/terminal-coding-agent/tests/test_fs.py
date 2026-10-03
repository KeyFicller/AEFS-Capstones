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


def test_read_file_survives_non_utf8(tmp_path: Path) -> None:
    """A binary file (e.g. MARC .mrc) must decode lossily instead of raising UnicodeDecodeError."""
    (tmp_path / "blob.bin").write_bytes(b"\x00\x95\xfe\xff")
    tool = build_read_file(tmp_path)
    assert tool.invoke({"path": "blob.bin"}) != ""


def test_edit_unique(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a x a\n", encoding="utf-8")
    tool = build_edit_file(tmp_path)
    out = tool.invoke({"path": "f.txt", "old_str": "x", "new_str": "y"})

    assert "y" in (tmp_path / "f.txt").read_text(encoding="utf-8")
    assert "---" in out or "@@" in out or "+y" in out


def test_edit_zero_matches_no_write(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("abc", encoding="utf-8")
    tool = build_edit_file(tmp_path)
    out = tool.invoke({"path": "f.txt", "old_str": "zzz", "new_str": "yyy"})

    assert out.startswith("Error:")
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "abc"


def test_edit_multi_matches_no_write(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x x", encoding="utf-8")
    tool = build_edit_file(tmp_path)
    out = tool.invoke({"path": "f.txt", "old_str": "x", "new_str": "y"})
    assert out.startswith("Error:")
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "x x"


def test_edit_empty_old_str_is_rejected(tmp_path: Path) -> None:
    """`"".count("")` is 1, so an empty needle would slip through on an empty file."""
    (tmp_path / "f.txt").write_text("", encoding="utf-8")
    tool = build_edit_file(tmp_path)
    out = tool.invoke({"path": "f.txt", "old_str": "", "new_str": "inserted"})

    assert out.startswith("Error:")
    assert "old_str" in out
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == ""


def test_edit_writes_atomically_and_keeps_mode(tmp_path: Path) -> None:
    target = tmp_path / "run.sh"
    target.write_text("echo old\n", encoding="utf-8")
    target.chmod(0o755)

    tool = build_edit_file(tmp_path)
    assert not tool.invoke({"path": "run.sh", "old_str": "old", "new_str": "new"}).startswith(
        "Error:"
    )

    assert target.read_text(encoding="utf-8") == "echo new\n"
    assert target.stat().st_mode & 0o777 == 0o755
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.sh"], "no temp file left behind"


def test_edit_escape_is_not_file_not_found(tmp_path: Path) -> None:
    tool = build_edit_file(tmp_path)
    out = tool.invoke({"path": "../outside.txt", "old_str": "a", "new_str": "b"})
    assert out.startswith("Error:")
    assert "escapes" in out.lower()
    assert "file not found" not in out.lower()
