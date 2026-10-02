"""The OCR-first baseline is the control condition, so these tests pin its contract.

Two properties decide whether the headline comparison means anything. Extraction
must prefer the PDF's own text layer and only fall back to Tesseract, and chunking
must be lossless: a chunker that splits ``16.8%`` into ``16.`` and ``8%`` destroys
ground truth the baseline would otherwise find, and would make the visual path look
better than it is.
"""

import pymupdf
import pytest
import torch
from langchain_core.documents import Document
from PIL import Image

from multimodal_doc_qa.baseline import ocr
from multimodal_doc_qa.retrievers.text import TextRetriever


# ------------------------------------------------------------------- chunking


def test_chunk_texts_is_lossless_across_a_boundary() -> None:
    """The tokens come back exactly, even where a fixed-width window would cut one."""
    text = " ".join(["aaaaaaaaaa"] * 39 + ["16.8%", "bbbbbbbbbb"])

    chunks = ocr.chunk_texts([text], max_chars=400)

    rebuilt = " ".join(chunk for chunk, _ in chunks)
    assert rebuilt.split() == text.split()


def test_chunk_texts_respects_max_chars() -> None:
    text = " ".join(f"tok{i}" for i in range(200))

    chunks = ocr.chunk_texts([text], max_chars=50)

    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk, _ in chunks)


def test_chunk_texts_keeps_an_oversized_token_intact() -> None:
    """Truncating it would drop content silently; emitting it whole must not crash."""
    long_token = "x" * 500

    chunks = ocr.chunk_texts([f"prefix {long_token} suffix"], max_chars=50)

    assert any(chunk == long_token for chunk, _ in chunks)


def test_chunk_texts_tags_each_chunk_with_its_page_index() -> None:
    chunks = ocr.chunk_texts(["alpha beta", "gamma delta"], max_chars=400)

    assert chunks == [("alpha beta", 0), ("gamma delta", 1)]


def test_chunk_texts_skips_pages_without_text() -> None:
    """A blank OCR result must not become an empty chunk that can never match."""
    chunks = ocr.chunk_texts(["", "   \n  ", "real text"], max_chars=400)

    assert chunks == [("real text", 2)]


def test_chunk_texts_of_nothing_is_nothing() -> None:
    assert ocr.chunk_texts([]) == []


# ----------------------------------------------------------------- extraction


def _pdf(tmp_path, page_texts: list[str]):
    """Write a PDF whose pages carry ``page_texts``; an empty string means no text layer."""
    path = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    for body in page_texts:
        page = doc.new_page()
        if body:
            page.insert_text((72, 72), body)
    doc.save(path)
    doc.close()
    return path


def _png(path, width: int = 10, height: int = 10):
    Image.new("RGB", (width, height), "white").save(path)
    return path


def test_extract_page_texts_reads_the_embedded_text_layer(tmp_path, monkeypatch) -> None:
    pdf = _pdf(tmp_path, ["Revenue 16.8%", "Costs 22.1%"])

    def _explode(*args, **kwargs):
        raise AssertionError("OCR must not run on a page that already has a text layer")

    monkeypatch.setattr(ocr.pytesseract, "image_to_string", _explode)

    # the image paths need not exist: a page with a text layer never touches them
    texts = ocr.extract_page_texts(pdf, [tmp_path / "p000.png", tmp_path / "p001.png"])

    assert "16.8%" in texts[0]
    assert "22.1%" in texts[1]


def test_extract_page_texts_falls_back_to_ocr_without_a_text_layer(tmp_path, monkeypatch) -> None:
    pdf = _pdf(tmp_path, [""])
    image = _png(tmp_path / "p000.png", width=10, height=10)
    seen: list[tuple[int, int]] = []

    def _fake_ocr(img, *args, **kwargs):
        seen.append(img.size)
        return "OCR TEXT 16.8%"

    monkeypatch.setattr(ocr.pytesseract, "image_to_string", _fake_ocr)

    texts = ocr.extract_page_texts(pdf, [image])

    assert texts == ["OCR TEXT 16.8%"]
    assert seen == [(10, 10)]


def test_extract_page_texts_pairs_each_page_with_its_own_image(tmp_path, monkeypatch) -> None:
    """Misaligned images would attribute one page's text to another."""
    pdf = _pdf(tmp_path, ["", ""])
    images = [_png(tmp_path / f"p{i:03d}.png", width=10 + i) for i in range(2)]

    monkeypatch.setattr(
        ocr.pytesseract, "image_to_string", lambda img, *a, **kw: str(img.size[0])
    )

    assert ocr.extract_page_texts(pdf, images) == ["10", "11"]


def test_extract_page_texts_rejects_a_mismatched_image_list(tmp_path) -> None:
    pdf = _pdf(tmp_path, ["a", "b"])

    with pytest.raises(ValueError):
        ocr.extract_page_texts(pdf, [tmp_path / "only-one.png"])


# ---------------------------------------------------------------- TextRetriever


class _SpyEmbedder:
    """Records every batch it is handed, and always returns the same unit vector."""

    def __init__(self) -> None:
        self.queries: list[list[str]] = []

    def encode(self, texts: list[str]) -> torch.Tensor:
        self.queries.append(texts)
        return torch.tensor([[1.0, 0.0, 0.0, 0.0]])


def _retriever(embedder: _SpyEmbedder | None = None, **kwargs: object) -> TextRetriever:
    """Chunk 0 answers the query direction; chunk 1 is orthogonal to it."""
    return TextRetriever(
        doc_id="doc000",
        embedder=embedder or _SpyEmbedder(),
        index=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        chunks=[("16.8%", 0), ("22.1%", 2)],
        **kwargs,
    )


def test_text_retriever_returns_chunks_in_score_order() -> None:
    docs = _retriever(k=2).invoke("16.8%")

    assert [d.page_content for d in docs] == ["16.8%", "22.1%"]


def test_text_retriever_ids_pages_where_the_renderer_wrote_them() -> None:
    """The id must resolve to an image: the renderer writes ``<doc>/p{page:03d}.png``.

    A bare ``p000`` would point at ``<render_dir>/p000.png``, which never exists, and
    would also alias page 0 of every other document in the pool.
    """
    docs = _retriever(k=2).invoke("16.8%")

    assert [d.metadata["page_id"] for d in docs] == ["doc000/p000", "doc000/p002"]


def test_text_retriever_reports_its_score() -> None:
    docs = _retriever(k=1).invoke("16.8%")

    assert docs[0].metadata["score"] == pytest.approx(1.0)


def test_text_retriever_honours_k() -> None:
    assert len(_retriever(k=1).invoke("16.8%")) == 1
    # the default k exceeds the two chunks, so both come back
    assert len(_retriever().invoke("16.8%")) == 2


def test_text_retriever_hands_the_raw_query_to_the_embedder() -> None:
    embedder = _SpyEmbedder()

    _retriever(embedder, k=1).invoke("what was the Q3 margin?")

    assert embedder.queries == [["what was the Q3 margin?"]]


def test_text_retriever_returns_langchain_documents() -> None:
    docs = _retriever(k=1).invoke("16.8%")

    assert all(isinstance(d, Document) for d in docs)


def test_text_retriever_over_an_empty_index_returns_nothing() -> None:
    retriever = TextRetriever(
        doc_id="doc000", embedder=_SpyEmbedder(), index=torch.empty((0, 4)), chunks=[], k=5
    )

    assert retriever.invoke("anything") == []
