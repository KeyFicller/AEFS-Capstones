"""REPL input shorthands: `@file`, `!shell`, `/command`.

`parse` is pure so the dispatch rules can be tested without a terminal.
"""

import base64
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage

_MENTION_RE = re.compile(r"@([^\s@]+)")


@dataclass(frozen=True)
class ParsedInput:
    """One REPL line, classified. `mentions` is only filled for `kind == "task"`."""

    kind: Literal["task", "shell", "command"]
    text: str
    command: str | None = None
    mentions: tuple[str, ...] = ()


def parse(line: str) -> ParsedInput:
    """Classify one line. `\\` at column 0 disables `!` and `/` for that line.

    `@` is still scanned. A bare `/` is a task because the command name is empty.
    """
    if line.startswith("\\"):
        body = line[1:]
        return ParsedInput(kind="task", text=body, mentions=_mentions(body))
    if line.startswith("/"):
        name, _, rest = line.partition(" ")
        if len(name) > 1:
            return ParsedInput(kind="command", text=rest.strip(), command=name[1:])
    if line.startswith("!"):
        return ParsedInput(kind="shell", text=line[1:].strip())
    return ParsedInput(kind="task", text=line, mentions=_mentions(line))


def _mentions(text: str) -> tuple[str, ...]:
    return tuple(_MENTION_RE.findall(text))


IMAGE_SUFFIXES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
_CHARS_PER_TOKEN = 4


class AttachmentError(Exception):
    """A `@path` could not be attached; the turn must not start."""


def build_task_message(
    parsed: ParsedInput,
    root: Path,
    extra_dirs: Sequence[Path] = (),
    *,
    token_limit: int = 4000,
) -> HumanMessage:
    """Append every `@path`. Search `root`, then each extra directory."""
    body = parsed.text
    images: list[dict] = []
    roots = (root, *extra_dirs)
    for mention in parsed.mentions:
        resolved = _resolve(roots, mention)
        body = body.replace(f"@{mention}", mention)
        appendix, blocks = _attachment(resolved, mention, token_limit)
        body += appendix
        images.extend(blocks)
    if not images:
        return HumanMessage(content=body)
    return HumanMessage(content=[{"type": "text", "text": body}, *images])


def _resolve(roots: Sequence[Path], mention: str) -> Path:
    if not mention.strip():
        raise AttachmentError("path is empty")
    escaped = False
    inside = False
    for base in roots:
        root = base.resolve()
        raw = Path(mention)
        target = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            escaped = True
            continue
        inside = True
        if target.is_file():
            return target
    if inside or not escaped:
        raise AttachmentError(f"not a file: {mention}")
    raise AttachmentError(f"path escapes directory: {mention}")


def _truncate(text: str, token_limit: int) -> str:
    max_chars = token_limit * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    marker = f"\n...[truncated, ~{token_limit} tokens]"
    keep = max(0, max_chars - len(marker))
    return text[:keep] + marker


def _attachment(path: Path, label: str, token_limit: int) -> tuple[str, list[dict]]:
    if (media_type := IMAGE_SUFFIXES.get(path.suffix.lower())) is not None:
        return "", [_image_block(path, label, media_type)]
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise AttachmentError(f"not a text or image file: {label}") from None
    except OSError as exc:
        raise AttachmentError(str(exc)) from None
    return f"\n\n[{label}]\n{_truncate(content, token_limit)}", []


def _image_block(path: Path, label: str, media_type: str) -> dict:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AttachmentError(str(exc)) from None
    if len(data) > MAX_IMAGE_BYTES:
        raise AttachmentError(
            f"image too large (> {MAX_IMAGE_BYTES // (1024 * 1024)} MB): {label}"
        )
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}}


MAX_COMPLETIONS = 200


def completion_candidates(
    text_before_cursor: str,
    roots: Sequence[Path],
    command_names: Sequence[str],
) -> list[str]:
    """`/` command names at column 0, or `@` paths under `roots` in order."""
    if text_before_cursor.startswith("/") and " " not in text_before_cursor:
        prefix = text_before_cursor[1:]
        return [f"/{name}" for name in command_names if name.startswith(prefix)]
    if "@" in text_before_cursor:
        prefix = text_before_cursor.rsplit("@", 1)[1]
        if " " in prefix:
            return []
        return [f"@{path}" for path in _paths(roots) if path.startswith(prefix)]
    return []


def _paths(roots: Sequence[Path]) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for base in roots:
        root = base.resolve()
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or ".git" in path.parts:
                continue
            relative = str(path.relative_to(root))
            if relative in seen:
                continue
            seen.add(relative)
            paths.append(relative)
            if len(paths) >= MAX_COMPLETIONS:
                return paths
    return paths
