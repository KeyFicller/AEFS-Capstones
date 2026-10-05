"""@ / ! / parsing: pure, no I/O."""

import base64

import pytest
from repl_console.shorthands import (
    AttachmentError,
    build_task_message,
    completion_candidates,
    parse,
)


def test_plain_line_is_a_task() -> None:
    parsed = parse("fix the login bug")
    assert parsed.kind == "task"
    assert parsed.text == "fix the login bug"
    assert parsed.command is None
    assert parsed.mentions == ()


def test_slash_at_line_start_is_a_command() -> None:
    parsed = parse("/review src/a.py")
    assert parsed.kind == "command"
    assert parsed.command == "review"
    assert parsed.text == "src/a.py"


def test_a_bare_slash_is_a_task() -> None:
    parsed = parse("/")
    assert parsed.kind == "task"
    assert parsed.text == "/"


def test_bang_at_line_start_is_shell() -> None:
    parsed = parse("!pytest -q")
    assert parsed.kind == "shell"
    assert parsed.text == "pytest -q"


def test_an_empty_bang_is_still_shell() -> None:
    assert parse("!").kind == "shell"
    assert parse("!").text == ""


def test_backslash_disables_bang_and_slash_only() -> None:
    assert parse("\\!ls").kind == "task"
    assert parse("\\!ls").text == "!ls"
    escaped = parse("\\@notes.txt")
    assert escaped.kind == "task"
    assert escaped.text == "@notes.txt"
    assert escaped.mentions == ("notes.txt",)


def test_slash_and_bang_below_the_first_line_are_plain_text() -> None:
    assert parse("explain a/b").kind == "task"
    assert parse("why not use a! b").kind == "task"


def test_mentions_are_collected_from_anywhere() -> None:
    parsed = parse("compare @a.txt with @dir/b.txt please")
    assert parsed.kind == "task"
    assert parsed.mentions == ("a.txt", "dir/b.txt")
    assert parsed.text == "compare @a.txt with @dir/b.txt please"


def test_missing_file_raises(tmp_path) -> None:
    with pytest.raises(AttachmentError, match="nope.txt"):
        build_task_message(parse("look at @nope.txt"), tmp_path)


def test_escaping_every_directory_raises(tmp_path) -> None:
    with pytest.raises(AttachmentError, match="escapes directory"):
        build_task_message(parse("read @../secret.txt"), tmp_path)


def test_a_directory_is_rejected(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    with pytest.raises(AttachmentError, match="not a file"):
        build_task_message(parse("read @src"), tmp_path)


def test_text_file_content_is_attached_and_path_is_kept(tmp_path) -> None:
    (tmp_path / "NOTES.md").write_text("hello", encoding="utf-8")
    message = build_task_message(parse("summarize @NOTES.md"), tmp_path)
    assert isinstance(message.content, str)
    assert "summarize NOTES.md" in message.content
    assert "[NOTES.md]\nhello" in message.content


def test_a_large_text_file_is_truncated(tmp_path) -> None:
    (tmp_path / "big.txt").write_text("x" * 20_000, encoding="utf-8")
    message = build_task_message(parse("look at @big.txt"), tmp_path, token_limit=4)
    assert "truncated" in message.content
    assert len(message.content) < 20_000


def test_an_image_becomes_a_data_url_block(tmp_path) -> None:
    payload = b"\x89PNG\r\n\x1a\nfake"
    (tmp_path / "shot.png").write_bytes(payload)
    message = build_task_message(parse("what is @shot.png"), tmp_path)
    assert isinstance(message.content, list)
    text_part, image_part = message.content
    assert text_part == {"type": "text", "text": "what is shot.png"}
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert base64.b64decode(image_part["image_url"]["url"].split(",", 1)[1]) == payload


def test_text_and_image_mentions_mix(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.jpg").write_bytes(b"jpegbytes")
    message = build_task_message(parse("compare @a.txt @b.jpg"), tmp_path)
    assert isinstance(message.content, list)
    assert "[a.txt]\nalpha" in message.content[0]["text"]
    assert message.content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_an_oversized_image_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("repl_console.shorthands.MAX_IMAGE_BYTES", 4)
    (tmp_path / "big.png").write_bytes(b"12345")
    with pytest.raises(AttachmentError, match="too large"):
        build_task_message(parse("@big.png"), tmp_path)


def test_an_unsupported_binary_extension_is_rejected(tmp_path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(AttachmentError, match="not a text or image file"):
        build_task_message(parse("@blob.bin"), tmp_path)


def test_extra_dir_supplies_a_file_the_root_lacks(tmp_path) -> None:
    root = tmp_path / "root"
    extra = tmp_path / "extra"
    root.mkdir()
    extra.mkdir()
    (extra / "page.txt").write_text("from-extra", encoding="utf-8")
    message = build_task_message(parse("read @page.txt"), root, [extra])
    assert "[page.txt]\nfrom-extra" in message.content


def test_root_wins_when_the_relative_path_exists_in_both(tmp_path) -> None:
    root = tmp_path / "root"
    extra = tmp_path / "extra"
    root.mkdir()
    extra.mkdir()
    (root / "page.txt").write_text("from-root", encoding="utf-8")
    (extra / "page.txt").write_text("from-extra", encoding="utf-8")
    message = build_task_message(parse("@page.txt"), root, [extra])
    assert "from-root" in message.content
    assert "from-extra" not in message.content


def test_a_path_that_escapes_the_root_can_still_hit_an_extra_dir(tmp_path) -> None:
    root = tmp_path / "root"
    extra = tmp_path / "extra"
    root.mkdir()
    extra.mkdir()
    (extra / "secret.txt").write_text("ok", encoding="utf-8")
    message = build_task_message(
        parse(f"read @{extra / 'secret.txt'}"),
        root,
        [extra],
    )
    assert "ok" in message.content


def test_completion_offers_slash_commands_at_line_start(tmp_path) -> None:
    assert completion_candidates("/he", [tmp_path], ("help", "quit")) == ["/help"]


def test_completion_offers_paths_after_at(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x", encoding="utf-8")
    (tmp_path / "readme.md").write_text("x", encoding="utf-8")
    assert completion_candidates("see @src/m", [tmp_path], ()) == ["@src/main.py"]


def test_completion_is_empty_without_a_sigil(tmp_path) -> None:
    assert completion_candidates("plain text", [tmp_path], ("help",)) == []


def test_completion_skips_a_relative_path_already_offered_by_an_earlier_root(tmp_path) -> None:
    root = tmp_path / "root"
    extra = tmp_path / "extra"
    root.mkdir()
    extra.mkdir()
    (root / "a.txt").write_text("x", encoding="utf-8")
    (extra / "a.txt").write_text("y", encoding="utf-8")
    (extra / "only-extra.txt").write_text("z", encoding="utf-8")
    assert completion_candidates("@", [root, extra], ()) == ["@a.txt", "@only-extra.txt"]


def test_completion_skips_git_directories(tmp_path) -> None:
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text("x", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    assert completion_candidates("@", [tmp_path], ()) == ["@keep.txt"]
