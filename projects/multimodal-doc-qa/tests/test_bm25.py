"""BM25 over the OCR chunks: the arithmetic against a naive double loop, and the contract.

The retriever must emit the same Document shape as ``TextRetriever`` so the agent,
citation parser and viewer need no change.
"""

import math

import pytest
from langchain_core.documents import Document

from multimodal_doc_qa.retrievers.bm25 import BM25, BM25Retriever, tokenize


def test_tokenize_lowercases_and_keeps_digits() -> None:
    assert tokenize("EMEA margin was 16.8%") == ["emea", "margin", "was", "16", "8"]


def test_tokenize_keeps_an_alphanumeric_id_whole() -> None:
    assert tokenize("P002, fig-3") == ["p002", "fig", "3"]


def test_tokenize_ignores_punctuation_only() -> None:
    assert tokenize("... ---") == []


def _naive_scores(corpus: list[list[str]], query: list[str]) -> list[float]:
    """An independent BM25 written straight from the formula, no Counter."""
    k1, b = 1.5, 0.75
    n = len(corpus)
    avgdl = sum(len(doc) for doc in corpus) / n
    scores = []
    for doc in corpus:
        total = 0.0
        for term in set(query):
            frequency = doc.count(term)
            if not frequency:
                continue
            df = sum(1 for other in corpus if term in other)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            denominator = frequency + k1 * (1.0 - b + b * len(doc) / avgdl)
            total += idf * frequency * (k1 + 1.0) / denominator
        scores.append(total)
    return scores


def test_bm25_matches_the_naive_formula() -> None:
    corpus = [["a", "a", "b"], ["a", "c"], ["d", "d"]]
    bm25 = BM25(corpus)

    for query in (["a"], ["a", "d"], ["c", "b"]):
        expected = _naive_scores(corpus, query)
        assert [bm25.score(query, index) for index in range(len(corpus))] == pytest.approx(
            expected
        )


def test_a_document_without_the_term_scores_zero() -> None:
    bm25 = BM25([["a"], ["b"]])

    assert bm25.score(["a"], 1) == 0.0


def _retriever(**kwargs: object) -> BM25Retriever:
    corpus = [("a/p000", "the margin rose"), ("b/p000", "fees fees fees"), ("c/p000", "fees")]
    return BM25Retriever(corpus=corpus, **kwargs)


def test_retriever_ranks_by_bm25() -> None:
    """``b`` repeats the term on a longer document; it must outrank the one-token ``c``."""
    docs = _retriever(k=2).invoke("fees")

    assert [doc.metadata["page_id"] for doc in docs] == ["b/p000", "c/p000"]


def test_retriever_exposes_chunk_text_and_score() -> None:
    docs = _retriever(k=1).invoke("margin")

    assert docs[0].page_content == "the margin rose"
    assert docs[0].metadata["page_id"] == "a/p000"
    assert docs[0].metadata["score"] > 0


def test_retriever_returns_langchain_documents() -> None:
    assert all(isinstance(doc, Document) for doc in _retriever().invoke("fees"))


def test_retriever_returns_nothing_when_no_term_matches() -> None:
    """Every score is 0, so the best-ratio floor keeps nothing."""
    assert _retriever().invoke("zebra") == []
