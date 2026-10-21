"""The shape every arm hands the graph and the eval.

``page_content`` means three different things across the arms (page id, chunk text,
abstract text). Consumers read this instead, so no node has to know which arm ran.
"""

from __future__ import annotations

from langchain_core.documents import Document
from pydantic import BaseModel


class RetrievalHit(BaseModel):
    """One retrieved page. ``rank`` is 1-based within this arm's result."""

    page_id: str
    score: float
    rank: int
    snippet: str | None = None


def documents_to_hits(documents: list[Document]) -> list[RetrievalHit]:
    """Convert one arm's output. The only place a hit is built, so rank cannot drift."""
    hits: list[RetrievalHit] = []
    for rank, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page_id")
        if page is None:
            raise ValueError(f"retrieved document has no page_id: {doc.page_content[:80]!r}")
        score = doc.metadata.get("score")
        if score is None:
            raise ValueError(f"retrieved document has no score: {page}")
        hits.append(
            RetrievalHit(
                page_id=page,
                score=float(score),
                rank=rank,
                snippet=doc.metadata.get("snippet"),
            )
        )
    return hits
