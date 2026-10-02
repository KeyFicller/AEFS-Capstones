"""The retriever is where a text query meets the visual index.

What matters here is the contract, not the arithmetic (``test_maxsim.py`` owns
that): the retriever must hand the *raw query string* to the encoder instead of
encoding itself, and it must emit real LangChain ``Document``s whose metadata
carries the page id and score. Every downstream consumer -- the agent, the
citation parser, the viewer -- reads that shape.
"""

import pytest
import torch
from langchain_core.documents import Document

from multimodal_doc_qa.index.maxsim import MultiVectorIndex
from multimodal_doc_qa.retrievers.multivector import MultiVectorRetriever

QUERY_VEC = torch.tensor([[1.0, 0.0, 0.0, 0.0]])


class _SpyEncoder:
    """Records every query it is handed, and always returns the same unit vector."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode_query(self, text: str) -> torch.Tensor:
        self.queries.append(text)
        return QUERY_VEC


def _index() -> MultiVectorIndex:
    """Page ``a`` owns the query direction; page ``b`` is orthogonal to it."""
    idx = MultiVectorIndex(device="cpu")
    idx.add("a", torch.eye(4))
    idx.add("b", torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    return idx


def _retriever(encoder: _SpyEncoder | None = None, **kwargs: object) -> MultiVectorRetriever:
    return MultiVectorRetriever(encoder=encoder or _SpyEncoder(), index=_index(), **kwargs)


def test_retriever_returns_pages_ranked_by_maxsim() -> None:
    docs = _retriever(k=2).invoke("anything")

    assert [d.metadata["page_id"] for d in docs] == ["a", "b"]


def test_retriever_exposes_page_id_and_score_in_the_document() -> None:
    docs = _retriever(k=1).invoke("anything")

    assert docs[0].page_content == "a"
    assert docs[0].metadata["page_id"] == "a"
    assert docs[0].metadata["score"] == pytest.approx(1.0)


def test_retriever_hands_the_raw_query_to_the_encoder() -> None:
    """Encoding is the encoder's job: swapping encoders is how the ablation works."""
    encoder = _SpyEncoder()

    _retriever(encoder, k=1).invoke("a table of fees")

    assert encoder.queries == ["a table of fees"]


def test_retriever_honours_k() -> None:
    assert len(_retriever(k=1).invoke("anything")) == 1
    # the default k exceeds the two indexed pages, so both come back
    assert len(_retriever().invoke("anything")) == 2


def test_retriever_returns_langchain_documents() -> None:
    docs = _retriever(k=1).invoke("anything")

    assert all(isinstance(d, Document) for d in docs)


def test_retriever_over_an_empty_index_returns_nothing() -> None:
    retriever = MultiVectorRetriever(encoder=_SpyEncoder(), index=MultiVectorIndex(device="cpu"))

    assert retriever.invoke("anything") == []
