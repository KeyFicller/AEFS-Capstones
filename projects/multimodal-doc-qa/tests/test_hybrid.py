"""Hybrid arm: RRF over two rankings, and the retriever that fuses them.

Both sub-retrievers are stubs. What is under test is the fusion, not the arithmetic
of either arm.
"""

from langchain_core.documents import Document

from multimodal_doc_qa.retrievers.hybrid import (
    HybridRetriever,
    pages_in_rank_order,
    reciprocal_rank_fusion,
)


class _StubRetriever:
    """Returns the same fixed ranking for any query, in the given page order."""

    def __init__(self, page_ids: list[str]) -> None:
        self.page_ids = page_ids
        self.calls: list[str] = []

    def invoke(self, query: str) -> list[Document]:
        self.calls.append(query)
        return [
            Document(page_content=page_id, metadata={"page_id": page_id, "score": 1.0})
            for page_id in self.page_ids
        ]


def test_rrf_sums_the_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])

    assert fused["a"] == 1 / 61
    assert fused["b"] == 1 / 62 + 1 / 61
    assert fused["c"] == 1 / 62


def test_rrf_ignores_rankings_a_page_is_absent_from() -> None:
    assert set(reciprocal_rank_fusion([["a"], ["b"]])) == {"a", "b"}


def test_pages_in_rank_order_keeps_a_pages_best_rank() -> None:
    """A text arm emits several chunks of one page; only its first appearance counts."""
    docs = [
        Document(page_content="x", metadata={"page_id": "a"}),
        Document(page_content="y", metadata={"page_id": "b"}),
        Document(page_content="z", metadata={"page_id": "a"}),
    ]

    assert pages_in_rank_order(docs) == ["a", "b"]


def test_pages_in_rank_order_reads_the_page_id_metadata() -> None:
    docs = [Document(page_content="a/p000", metadata={"page_id": "a/p000"})]

    assert pages_in_rank_order(docs) == ["a/p000"]


def _hybrid(dense: list[str], lexical: list[str], **kwargs: object) -> HybridRetriever:
    return HybridRetriever(dense=_StubRetriever(dense), lexical=_StubRetriever(lexical), **kwargs)


def test_a_page_both_arms_rank_first_wins() -> None:
    docs = _hybrid(["a", "b"], ["a", "c"], k=3).invoke("q")

    assert [doc.metadata["page_id"] for doc in docs] == ["a", "c", "b"]


def test_a_page_only_one_arm_found_still_appears() -> None:
    docs = _hybrid(["a"], ["b"], k=3).invoke("q")

    assert {doc.metadata["page_id"] for doc in docs} == {"a", "b"}


def test_a_page_neither_arm_found_is_absent() -> None:
    docs = _hybrid(["a"], ["a"], k=3).invoke("q")

    assert [doc.metadata["page_id"] for doc in docs] == ["a"]


def test_hybrid_honours_k() -> None:
    docs = _hybrid(["a", "b", "c"], ["d"], k=2).invoke("q")

    assert len(docs) == 2


def test_hybrid_exposes_page_id_and_fused_score() -> None:
    docs = _hybrid(["a"], ["a"], k=1).invoke("q")

    assert docs[0].page_content == "a"
    assert docs[0].metadata["score"] == 2 / 61


def test_hybrid_hands_the_raw_query_to_both_arms() -> None:
    dense, lexical = _StubRetriever(["a"]), _StubRetriever(["b"])
    HybridRetriever(dense=dense, lexical=lexical, k=2).invoke("a table of fees")

    assert dense.calls == ["a table of fees"]
    assert lexical.calls == ["a table of fees"]
