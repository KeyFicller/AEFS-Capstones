import hashlib
from pathlib import Path

from multimodal_doc_qa.schemas import PdfDocument, TextDocument
from multimodal_doc_qa.summarize import index_summaries
from PIL import Image


def _png(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)


def test_identical_pages_are_described_once(tmp_path: Path) -> None:
    _png(tmp_path / "a" / "p000.png", "white")
    _png(tmp_path / "b" / "p000.png", "white")
    calls: list[Path] = []

    def describe(png: Path) -> str:
        calls.append(png)
        return "a blank page"

    cache: dict[str, str] = {}
    docs = [PdfDocument(doc_id="a"), PdfDocument(doc_id="b")]

    def encode(texts: list[str]):
        return texts

    first = index_summaries(docs, tmp_path, encode, cache, None, describe)
    second = index_summaries(docs, tmp_path, encode, cache, None, describe)

    assert len(calls) == 1
    assert len(cache) == 1
    assert first["a"]["chunks"] == [("a blank page", 0)]
    assert second["b"]["chunks"] == [("a blank page", 0)]
    digest = hashlib.sha256((tmp_path / "a" / "p000.png").read_bytes()).hexdigest()
    assert cache[digest] == "a blank page"


def test_a_text_page_is_its_own_description(tmp_path: Path) -> None:
    def describe(png: Path) -> str:
        raise AssertionError("text pages are not shown to the VLM")

    payload = index_summaries(
        [TextDocument(doc_id="note", pages=["The margin was 16.8%."])],
        tmp_path,
        lambda texts: texts,
        {},
        None,
        describe,
    )
    assert payload["note"]["chunks"] == [("The margin was 16.8%.", 0)]
