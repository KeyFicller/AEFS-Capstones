"""Visual retriever: query text, MaxSim over page images, LangChain documents.

``page_content`` is the page id. The page's content is the image, refetched by id.
"""

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
import torch

from multimodal_doc_qa.index.maxsim import MultiVectorIndex


class MultiVectorRetriever(BaseRetriever):
    """Late-interaction ranker. ``encoder`` is ``Any`` so tests can inject a stub."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    encoder: Any
    index: MultiVectorIndex
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Best ``k`` pages. ``page_content`` and ``metadata["page_id"]`` are the page id."""
        query_vec: torch.Tensor = self.encoder.encode_query(query)
        return [
            Document(
                page_content=hit.page_id,
                metadata={"page_id": hit.page_id, "score": hit.score}
            ) for hit in self.index.search(query_vec, k=self.k)
        ]
