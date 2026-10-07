"""Page descriptions for the abstract index. One VLM call per distinct image, then the cache."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from multimodal_doc_qa.schemas import Document, TextDocument


def load_cache(path: Path) -> dict[str, str]:
    """Read ``{sha256: description}``. A missing file is an empty cache."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path: Path, cache: dict[str, str]) -> None:
    """Write the cache. Called after every new description so a crash keeps the pages already paid for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def describe_page(model: object, png: Path) -> str:
    """Ask ``model`` for a retrieval description of one rendered page."""
    import base64

    from langchain_core.messages import HumanMessage

    data = base64.b64encode(png.read_bytes()).decode()
    reply = model.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Describe this document page for retrieval. "
                            "Include the names, numbers, and labels a question might ask for."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}},
                ]
            )
        ]
    )
    content = reply.content
    if isinstance(content, str):
        return content.strip()
    return "\n".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ).strip()


def index_abstracts(
    documents: list[Document],
    render_dir: Path,
    encode: Callable[[list[str]], object],
    cache: dict[str, str],
    cache_path: Path | None,
    describe: Callable[[Path], str],
) -> dict[str, dict]:
    """One embedded description per page.

    A text page is its own description. An image page is described by ``describe`` unless
    ``cache`` already holds the sha256 of that PNG. Identical bytes share one description.
    """
    payloads: dict[str, dict] = {}
    for document in documents:
        chunks = _chunks(document, render_dir, cache, cache_path, describe)
        if not chunks:
            continue
        payloads[document.doc_id] = {
            "index": encode([text for text, _ in chunks]),
            "chunks": chunks,
        }
    return payloads


def _chunks(
    document: Document,
    render_dir: Path,
    cache: dict[str, str],
    cache_path: Path | None,
    describe: Callable[[Path], str],
) -> list[tuple[str, int]]:
    if isinstance(document, TextDocument):
        return [(text, index) for index, text in enumerate(document.pages)]
    chunks: list[tuple[str, int]] = []
    for png in sorted((render_dir / document.doc_id).glob("p*.png")):
        key = hashlib.sha256(png.read_bytes()).hexdigest()
        text = cache.get(key)
        if text is None:
            text = describe(png)
            cache[key] = text
            if cache_path is not None:
                save_cache(cache_path, cache)
        chunks.append((text, int(png.stem[1:])))
    return chunks
