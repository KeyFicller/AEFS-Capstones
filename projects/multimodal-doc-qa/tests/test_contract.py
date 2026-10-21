"""One shape for every arm: page id, score, position, and the chunk text when there is one."""

import pytest
from langchain_core.documents import Document
from multimodal_doc_qa.retrievers.contract import RetrievalHit, documents_to_hits


def test_rank_is_one_based_and_follows_the_retriever_order() -> None:
    docs = [
        Document(page_content="x", metadata={"page_id": "a/p000", "score": 0.9}),
        Document(page_content="y", metadata={"page_id": "b/p000", "score": 0.8}),
    ]

    hits = documents_to_hits(docs)

    assert [(hit.page_id, hit.rank) for hit in hits] == [("a/p000", 1), ("b/p000", 2)]


def test_a_page_may_appear_twice() -> None:
    """The OCR arm retrieves chunks, and two chunks of one page are two hits. Deduping would
    change recall@5's denominator, so the contract keeps them."""
    docs = [
        Document(page_content="x", metadata={"page_id": "a/p000", "score": 0.9}),
        Document(page_content="y", metadata={"page_id": "a/p000", "score": 0.8}),
    ]

    assert [hit.rank for hit in documents_to_hits(docs)] == [1, 2]


def test_snippet_is_carried_when_the_arm_set_one() -> None:
    docs = [
        Document(page_content="chunk", metadata={"page_id": "a/p000", "score": 1.0, "snippet": "chunk"})
    ]

    assert documents_to_hits(docs)[0].snippet == "chunk"


def test_a_visual_hit_has_no_snippet() -> None:
    docs = [Document(page_content="a/p000", metadata={"page_id": "a/p000", "score": 1.0})]

    assert documents_to_hits(docs)[0].snippet is None


def test_a_missing_page_id_is_an_error() -> None:
    with pytest.raises(ValueError, match="page_id"):
        documents_to_hits([Document(page_content="x", metadata={"score": 0.5})])


def test_a_missing_score_is_an_error() -> None:
    with pytest.raises(ValueError, match="score"):
        documents_to_hits([Document(page_content="x", metadata={"page_id": "a/p000"})])


def test_hit_holds_a_float_score() -> None:
    docs = [Document(page_content="x", metadata={"page_id": "a/p000", "score": 1})]

    hit = documents_to_hits(docs)[0]

    assert isinstance(hit, RetrievalHit)
    assert isinstance(hit.score, float)
