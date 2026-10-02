"""OCR-first retriever. Same contract as the visual retriever; what is indexed is text chunks."""

from typing import Any

import torch
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from multimodal_doc_qa.schemas import page_id


class TextEmbedder:
    """Single-vector text embedder. The model import is deferred so importing this module stays cheap."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Return L2-normalized ``[len(texts), dim]`` embeddings."""
        return self._model.encode(texts, normalize_embeddings=True, convert_to_tensor=True)


class TextRetriever(BaseRetriever):
    """Dot-product ranker over one document's chunks.

    Row ``i`` of ``index`` is ``chunks[i]``. ``doc_id`` is required: page 0 exists in every document.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    doc_id: str
    embedder: Any
    index: torch.Tensor
    chunks: list[tuple[str, int]]
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Best ``k`` chunks. ``page_content`` is the chunk text; ``metadata`` has ``page_id`` and ``score``."""
        query_vec = self.embedder.encode([query])[0]
        scores = self.index @ query_vec
        order = torch.argsort(scores, descending=True)[: self.k]
        return [
            Document(
                page_content=self.chunks[i][0],
                metadata={
                    "page_id": page_id(self.doc_id, self.chunks[i][1]),
                    "score": float(scores[i]),
                },
            )
            for i in order.tolist()
        ]


class MultiDocTextRetriever(BaseRetriever):
    """Corpus-wide top ``k`` by merging per-document retrievers.

    A chunk outside its own document's top ``k`` cannot be in the global top ``k``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    retrievers: list[TextRetriever]
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Return the corpus-wide best ``k`` chunks, best first."""
        merged = [doc for retriever in self.retrievers for doc in retriever.invoke(query)]
        merged.sort(key=lambda doc: -doc.metadata["score"])
        return merged[: self.k]
