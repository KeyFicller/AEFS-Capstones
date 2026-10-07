"""Hybrid arm: fuse the lexical ranking with one semantic ranking by reciprocal rank fusion.

Ranks, not scores: the two arms score on different scales, so RRF needs no calibration.
"""

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

RRF_K = 60


def pages_in_rank_order(documents: list[Document]) -> list[str]:
    """Page ids in the ranking's own order, one per page. A page's best rank is its first."""
    return list(dict.fromkeys(doc.metadata["page_id"] for doc in documents))


def reciprocal_rank_fusion(rankings: list[list[str]], *, rrf_k: int = RRF_K) -> dict[str, float]:
    """``sum over rankings of 1 / (rrf_k + rank)``, ``rank`` 1-based within each ranking."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, page_id in enumerate(ranking, start=1):
            scores[page_id] = scores.get(page_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores


class HybridRetriever(BaseRetriever):
    """Fuse ``lexical`` and ``dense`` by RRF, both queried deeper than ``k`` by the caller.

    ``page_content`` is the page id, matching the visual retriever. RRF scores are not
    similarities, so no ``min_score_ratio`` is applied after fusion.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dense: Any
    lexical: Any
    k: int = 5
    rrf_k: int = RRF_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Top ``k`` pages by fused rank."""
        rankings = [
            pages_in_rank_order(self.lexical.invoke(query)),
            pages_in_rank_order(self.dense.invoke(query)),
        ]
        fused = reciprocal_rank_fusion(rankings, rrf_k=self.rrf_k)
        ordered = sorted(fused, key=lambda page_id: -fused[page_id])[: self.k]
        return [
            Document(page_content=page_id, metadata={"page_id": page_id, "score": fused[page_id]})
            for page_id in ordered
        ]
