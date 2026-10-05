"""Image blocks stay on the executor message and are stripped for planner and summary."""

from langchain_core.messages import AIMessage, HumanMessage
from terminal_coding_agent import attachments


def test_strip_images_keeps_only_text_parts() -> None:
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
    )
    stripped = attachments.strip_images([message, AIMessage(content="ok")])

    assert stripped[0].content == "describe this"
    assert stripped[1].content == "ok"


def test_attached_images_reads_the_latest_human_message() -> None:
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    messages = [HumanMessage(content=[{"type": "text", "text": "hi"}, image])]
    assert attachments.attached_images(messages) == [image]
    assert attachments.attached_images([HumanMessage(content="plain")]) == []
