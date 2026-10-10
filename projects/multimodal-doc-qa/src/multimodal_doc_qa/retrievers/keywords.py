"""The ``lexical-kw`` arm: an LLM extracts the query keywords, BM25 ranks on the result.

The extraction is one structured call per query. The chat model is built on the first call
and reused, so preloading every mode does not load a model for this one arm.
"""

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, ConfigDict, Field

_KEYWORDS_SYS = (
    "Extract the search keywords from the question. Return only the content words a "
    "document would use: names, numbers, and terms. Drop the question words and the "
    "filler. Do not answer the question."
)


class Keywords(BaseModel):
    """The query's keywords, as the extractor model returns them."""

    keywords: list[str] = Field(default_factory=list)


class KeywordExtractor:
    """``query -> keywords``. The model bound to ``Keywords`` is built on the first call, then reused.

    The model comes from the host's single construction point, ``build_chat_model``.
    """

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._bound: Any = None

    def __call__(self, query: str) -> list[str]:
        """The question's keywords, blanks dropped.

        Duplicates are left alone: BM25 scores ``set(query)``, so a repeated keyword changes
        no score.
        """
        if self._bound is None:
            from multimodal_doc_qa.synth.answer import build_chat_model

            self._bound = build_chat_model(self._settings).with_structured_output(Keywords)
        result = self._bound.invoke(
            [SystemMessage(content=_KEYWORDS_SYS), HumanMessage(content=query)]
        )
        return [keyword for keyword in (raw.strip() for raw in result.keywords) if keyword]


class KeywordRetriever(BaseRetriever):
    """Rank with ``inner``, but on the extractor's keywords instead of the raw query.

    A query the extractor reduces to nothing falls back to the raw query: an empty query
    scores every document at zero, so it could only return nothing.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: Any
    extractor: Any

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """The inner retriever's hits for ``query``'s keywords."""
        keywords = self.extractor(query)
        return self.inner.invoke(" ".join(keywords) or query)
