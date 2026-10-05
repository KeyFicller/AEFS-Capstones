"""Image blocks on a turn's message. Kept here so the graph never imports the REPL."""

from __future__ import annotations

from langchain_core.messages import HumanMessage


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
