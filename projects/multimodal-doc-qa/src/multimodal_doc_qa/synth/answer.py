"""Answer synthesis. Citations come from structured output; a reply that misses ``Answer`` raises.

``Citation.bbox`` may be omitted. A malformed box may not.
"""

import base64
from pathlib import Path

from multimodal_doc_qa.config import Settings
from multimodal_doc_qa.schemas import Answer

_SYSTEM = (
    "Answer ONLY from the provided page images. Cite every claim with the doc_id and "
    "the page you read it from. Include a bbox normalized to 0..1 when you can localize "
    "the claim (x0, y0 top-left; x1, y1 bottom-right); omit bbox when you cannot."
)


def _data_url(png: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()


def build_page_blocks(page_ids: list[str], render_dir: Path) -> list[dict]:
    """Image content parts, each preceded by a text label of its page id.

    Without the label the model cannot cite a page back. The image is a data URL; a file path is useless to a hosted model.
    """
    return [
        block
        for pid in page_ids
        for block in (
            {"type": "text", "text": f"page {pid}"},
            {"type": "image_url", "image_url": {"url": _data_url(render_dir / f"{pid}.png")}},
        )
    ]


def build_chat_model(settings: Settings):
    """The answerer chat model, with thinking mode off.

    Thinking mode rejects the forced tool choice that ``with_structured_output`` sets.
    Every structured caller must come through here.
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        settings.answerer_model,
        model_provider=settings.answerer_provider,
        extra_body={"thinking": {"type": "disabled"}},
    )


class AnswerSynthesizer:
    """Hosted VLM bound to ``Answer``. A reply that misses the schema raises."""

    def __init__(self, settings: Settings) -> None:
        self._model = build_chat_model(settings)
        self._structured = self._model.with_structured_output(Answer)

    def synthesize(self, question: str, page_ids: list[str], render_dir: Path) -> Answer:
        """Answer ``question`` from the rendered pages. A reply that misses ``Answer`` raises."""
        content = [{"type": "text", "text": question}, *build_page_blocks(page_ids, render_dir)]
        return self._structured.invoke([
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": content},
        ])