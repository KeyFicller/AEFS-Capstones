"""`@` / `!` / `/` parsing: pure, no I/O."""

import base64

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from terminal_coding_agent import shorthands


def test_plain_line_is_a_task() -> None:
    parsed = shorthands.parse("fix the login bug")
    assert parsed.kind == "task"
    assert parsed.text == "fix the login bug"
    assert parsed.command is None
    assert parsed.mentions == ()


def test_slash_at_line_start_is_a_command() -> None:
    parsed = shorthands.parse("/review src/a.py")
    assert parsed.kind == "command"
    assert parsed.command == "review"
    assert parsed.text == "src/a.py"


def test_bang_at_line_start_is_shell() -> None:
    parsed = shorthands.parse("!pytest -q")
    assert parsed.kind == "shell"
    assert parsed.text == "pytest -q"


def test_backslash_escapes_the_prefix() -> None:
    parsed = shorthands.parse("\\!ls")
    assert parsed.kind == "task"
    assert parsed.text == "!ls"


def test_slash_and_bang_below_the_first_line_are_plain_text() -> None:
    assert shorthands.parse("explain a/b").kind == "task"
    assert shorthands.parse("why not use a! b").kind == "task"


def test_mentions_are_collected_from_anywhere() -> None:
    parsed = shorthands.parse("compare @a.txt with @dir/b.txt please")
    assert parsed.kind == "task"
    assert parsed.mentions == ("a.txt", "dir/b.txt")
    assert parsed.text == "compare @a.txt with @dir/b.txt please"


def test_missing_file_raises_and_is_not_attached(tmp_path) -> None:
    parsed = shorthands.parse("look at @nope.txt")
    with pytest.raises(shorthands.AttachmentError, match="nope.txt"):
        shorthands.build_task_message(parsed, tmp_path)


def test_escaping_the_worktree_raises(tmp_path) -> None:
    parsed = shorthands.parse("read @../secret.txt")
    with pytest.raises(shorthands.AttachmentError, match="escapes worktree"):
        shorthands.build_task_message(parsed, tmp_path)


def test_a_directory_is_rejected(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    parsed = shorthands.parse("read @src")
    with pytest.raises(shorthands.AttachmentError, match="not a file"):
        shorthands.build_task_message(parsed, tmp_path)


def test_text_file_content_is_attached_and_path_is_kept(tmp_path) -> None:
    (tmp_path / "NOTES.md").write_text("hello", encoding="utf-8")
    parsed = shorthands.parse("summarize @NOTES.md")
    message = shorthands.build_task_message(parsed, tmp_path)

    assert isinstance(message.content, str)
    assert "summarize NOTES.md" in message.content  # path kept, sigil dropped
    assert "[NOTES.md]\nhello" in message.content


def test_a_large_text_file_is_truncated(tmp_path) -> None:
    (tmp_path / "big.txt").write_text("x" * 10_000_000, encoding="utf-8")
    parsed = shorthands.parse("look at @big.txt")
    message = shorthands.build_task_message(parsed, tmp_path)
    assert "truncated" in message.content


def test_strip_images_keeps_only_text_parts() -> None:
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
    )
    stripped = shorthands.strip_images([message, AIMessage(content="ok")])

    assert stripped[0].content == "describe this"
    assert stripped[1].content == "ok"


def test_attached_images_reads_the_latest_human_message() -> None:
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    messages = [HumanMessage(content=[{"type": "text", "text": "hi"}, image])]
    assert shorthands.attached_images(messages) == [image]
    assert shorthands.attached_images([HumanMessage(content="plain")]) == []


def test_an_image_becomes_a_data_url_block(tmp_path) -> None:
    payload = b"\x89PNG\r\n\x1a\nfake"
    (tmp_path / "shot.png").write_bytes(payload)

    message = shorthands.build_task_message(shorthands.parse("what is @shot.png"), tmp_path)

    assert isinstance(message.content, list)
    text_part, image_part = message.content
    assert text_part == {"type": "text", "text": "what is shot.png"}
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert base64.b64decode(image_part["image_url"]["url"].split(",", 1)[1]) == payload


def test_text_and_image_mentions_mix(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.jpg").write_bytes(b"jpegbytes")

    message = shorthands.build_task_message(shorthands.parse("compare @a.txt @b.jpg"), tmp_path)

    assert isinstance(message.content, list)
    assert "[a.txt]\nalpha" in message.content[0]["text"]
    assert message.content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_an_oversized_image_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shorthands, "MAX_IMAGE_BYTES", 4)
    (tmp_path / "big.png").write_bytes(b"12345")

    with pytest.raises(shorthands.AttachmentError, match="too large"):
        shorthands.build_task_message(shorthands.parse("@big.png"), tmp_path)


def test_an_unsupported_binary_extension_is_rejected(tmp_path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00")  # invalid UTF-8: not decodable text
    with pytest.raises(shorthands.AttachmentError, match="not a text or image file"):
        shorthands.build_task_message(shorthands.parse("@blob.bin"), tmp_path)




def test_completion_offers_slash_commands_at_line_start(tmp_path) -> None:
    assert shorthands.completion_candidates("/he", tmp_path, ("help", "quit")) == ["/help"]


def test_completion_offers_worktree_paths_after_at(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x", encoding="utf-8")
    (tmp_path / "readme.md").write_text("x", encoding="utf-8")

    assert shorthands.completion_candidates("see @src/m", tmp_path, ()) == ["@src/main.py"]


def test_completion_is_empty_without_a_sigil(tmp_path) -> None:
    assert shorthands.completion_candidates("plain text", tmp_path, ("help",)) == []
