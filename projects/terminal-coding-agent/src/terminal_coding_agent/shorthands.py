"""REPL input shorthands: `@file`, `!shell`, `/command`.

`parse` is pure so the dispatch rules can be tested without a terminal.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage

from terminal_coding_agent.tools.path import resolve_in_worktree
from terminal_coding_agent.tools.truncate import truncate

_MENTION_RE = re.compile(r"@([^\s@]+)")

IMAGE_SUFFIXES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
MAX_IMAGE_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class ParsedInput:
    """One REPL line, classified. `mentions` is only filled for `kind == "task"`."""

    kind: Literal["task", "shell", "command"]
    text: str
    command: str | None = None
    mentions: tuple[str, ...] = ()


def parse(line: str) -> ParsedInput:
    """Classify one line. `\\` at column 0 disables shorthand recognition for the line."""
    if line.startswith("\\"):
        return ParsedInput(kind="task", text=line[1:], mentions=_mentions(line[1:]))
    if line.startswith("/"):
        name, _, rest = line.partition(" ")
        if len(name) > 1:
            return ParsedInput(kind="command", text=rest.strip(), command=name[1:])
    if line.startswith("!"):
        return ParsedInput(kind="shell", text=line[1:].strip())
    return ParsedInput(kind="task", text=line, mentions=_mentions(line))


def _mentions(text: str) -> tuple[str, ...]:
    return tuple(_MENTION_RE.findall(text))


class AttachmentError(Exception):
    """A `@path` could not be attached; the turn must not start."""


def build_task_message(parsed: ParsedInput, worktree: Path) -> HumanMessage:
    """Append every `@path`'s content to the turn's message; raise before the graph runs."""
    body = parsed.text
    images: list[dict] = []
    for mention in parsed.mentions:
        resolved = resolve_in_worktree(worktree, mention)
        if isinstance(resolved, str):
            raise AttachmentError(resolved)
        body = body.replace(f"@{mention}", mention)
        appendix, blocks = _attachment(resolved, mention)
        body += appendix
        images.extend(blocks)
    if not images:
        return HumanMessage(content=body)
    return HumanMessage(content=[{"type": "text", "text": body}, *images])


def _attachment(path: Path, label: str) -> tuple[str, list[dict]]:
    """(text appendix, image blocks) for one mention."""
    if not path.is_file():
        raise AttachmentError(f"Error: not a file: {label}")
    if (media_type := IMAGE_SUFFIXES.get(path.suffix.lower())) is not None:
        return "", [_image_block(path, label, media_type)]
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise AttachmentError(f"Error: not a text or image file: {label}") from None
    except OSError as exc:
        raise AttachmentError(f"Error: {exc}") from None
    return f"\n\n[{label}]\n{truncate(content)}", []


def _image_block(path: Path, label: str, media_type: str) -> dict:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AttachmentError(f"Error: {exc}") from None
    if len(data) > MAX_IMAGE_BYTES:
        raise AttachmentError(
            f"Error: image too large (> {MAX_IMAGE_BYTES // (1024 * 1024)} MB): {label}"
        )
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}}


def strip_images(messages: list) -> list:
    """Copies of `messages` with image blocks dropped; text parts are kept verbatim."""
    stripped = []
    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(content, list):
            text = "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
            stripped.append(message.model_copy(update={"content": text}))
        else:
            stripped.append(message)
    return stripped


def attached_images(messages: list) -> list[dict]:
    """Image blocks of the most recent multimodal human message; [] when there are none."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, list):
            return [
                part
                for part in message.content
                if isinstance(part, dict) and part.get("type") == "image_url"
            ]
    return []


MAX_COMPLETIONS = 200


def completion_candidates(
    text_before_cursor: str, worktree: Path, command_names: Sequence[str]
) -> list[str]:
    """Completions for the token at the cursor: `/` command names, `@` worktree paths."""
    if text_before_cursor.startswith("/") and " " not in text_before_cursor:
        prefix = text_before_cursor[1:]
        return [f"/{name}" for name in command_names if name.startswith(prefix)]
    if "@" in text_before_cursor:
        prefix = text_before_cursor.rsplit("@", 1)[1]
        if " " in prefix:
            return []
        return [f"@{path}" for path in _worktree_paths(worktree) if path.startswith(prefix)]
    return []


def _worktree_paths(worktree: Path) -> list[str]:
    root = worktree.resolve()
    paths = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            paths.append(str(path.relative_to(root)))
            if len(paths) >= MAX_COMPLETIONS:
                break
    return paths
