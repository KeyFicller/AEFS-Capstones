from pathlib import Path

from langchain_core.documents import Document
from multimodal_doc_qa.retrievers.rerank import PageRank, order_documents, rerank_pages
from PIL import Image


def _doc(page_id: str) -> Document:
    return Document(page_content=page_id, metadata={"page_id": page_id, "score": 1.0})


def test_order_documents_uses_the_model_order_and_keeps_omissions() -> None:
    docs = [_doc("a/p000"), _doc("b/p000"), _doc("c/p000")]
    ordered = order_documents(docs, ["c/p000", "a/p000"])
    assert [doc.metadata["page_id"] for doc in ordered] == ["c/p000", "a/p000", "b/p000"]


def test_rerank_skips_the_model_for_a_single_page() -> None:
    class _Model:
        def invoke(self, messages: object) -> PageRank:
            raise AssertionError("one page has nothing to reorder")

    docs = [_doc("a/p000")]
    assert rerank_pages(_Model(), "q", docs, Path(), None) is docs


def test_rerank_follows_the_model(tmp_path: Path) -> None:
    for name in ("p000.png", "p001.png"):
        folder = tmp_path / "d"
        folder.mkdir(exist_ok=True)
        Image.new("RGB", (2, 2), "white").save(folder / name)

    class _Model:
        def invoke(self, messages: object) -> PageRank:
            return PageRank(page_ids=["d/p001", "d/p000"])

    ordered = rerank_pages(
        _Model(),
        "where is the total",
        [_doc("d/p000"), _doc("d/p001")],
        tmp_path,
        None,
    )
    assert [doc.metadata["page_id"] for doc in ordered] == ["d/p001", "d/p000"]
