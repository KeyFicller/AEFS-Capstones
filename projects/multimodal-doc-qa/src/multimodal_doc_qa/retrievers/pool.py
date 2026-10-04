"""Whole-page retriever: mean-pool the multi-vector, then one dot product per page."""

from typing import Any

import torch
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from multimodal_doc_qa.retrievers import within_best_ratio


def mean_vector(matrix: torch.Tensor) -> torch.Tensor:
    """Average the rows of a ``[n, dim]`` multi-vector and L2-normalize the result.

    A zero vector stays zero. Normalizing keeps the score a cosine, so the same
    ``min_score_ratio`` cutoff used by the other arms still means "this close to the best".
    """
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"expected a [n, dim] matrix with n > 0, got shape {tuple(matrix.shape)}")
    pooled = matrix.detach().float().mean(dim=0)
    norm = pooled.norm()
    if float(norm) == 0.0:
        return pooled
    return pooled / norm


class PooledRetriever(BaseRetriever):
    """Rank pages by the cosine of their mean-pooled patch vectors.

    ``encoder.encode_query`` still returns a multi-vector; this retriever pools it the
    same way the pages were pooled at load time.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    encoder: Any
    pages: dict[str, torch.Tensor]
    k: int = 5
    min_score_ratio: float = 0.5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Best ``k`` pages. ``page_content`` and ``metadata["page_id"]`` are the page id."""
        query_vec = mean_vector(self.encoder.encode_query(query))
        scored = [
            (page_id, float(vector @ query_vec)) for page_id, vector in self.pages.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        hits = [
            Document(page_content=page_id, metadata={"page_id": page_id, "score": score})
            for page_id, score in scored[: self.k]
        ]
        return within_best_ratio(hits, self.min_score_ratio)
