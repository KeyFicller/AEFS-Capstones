"""Lexical arm: BM25 over the OCR chunks. Hand-written so the ablation adds no dependency."""

from __future__ import annotations

import math
import re
from collections import Counter

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

from multimodal_doc_qa.retrievers import within_best_ratio

_SPLIT = re.compile(r"[^0-9a-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on anything that is not a digit or a letter.

    Digits survive so ``16.8%`` and ``p002`` stay matchable; a punctuation-only
    string yields no tokens.
    """
    return [token for token in _SPLIT.split(text.lower()) if token]


class BM25:
    """Corpus statistics for BM25. ``k1`` and ``b`` are the standard defaults."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        """Precompute document frequency, lengths and the mean length once."""
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n = len(corpus)
        self.lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.lengths) / self.n if self.n else 0.0
        self.tf = [Counter(doc) for doc in corpus]
        self.df = Counter(term for doc in corpus for term in set(doc))

    def score(self, query: list[str], index: int) -> float:
        """BM25 of ``query`` against the document at ``index``. 0 when ``avgdl`` is 0."""
        if self.avgdl == 0.0:
            return 0.0
        length = self.lengths[index]
        total = 0.0
        for term in set(query):
            frequency = self.tf[index].get(term, 0)
            if frequency == 0:
                continue
            df = self.df[term]
            idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)
            denominator = frequency + self.k1 * (1.0 - self.b + self.b * length / self.avgdl)
            total += idf * frequency * (self.k1 + 1.0) / denominator
        return total


class BM25Retriever(BaseRetriever):
    """Rank chunks by BM25. ``corpus`` is one ``(page_id, text)`` row per chunk."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    corpus: list[tuple[str, str]]
    k: int = 5
    min_score_ratio: float = 0.5
    _bm25: BM25 = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._bm25 = BM25([tokenize(text) for _, text in self.corpus])

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Best ``k`` chunks. Same Document shape as ``TextRetriever``."""
        tokens = tokenize(query)
        scores = [self._bm25.score(tokens, index) for index in range(len(self.corpus))]
        order = sorted(range(len(scores)), key=lambda index: -scores[index])[: self.k]
        hits = [
            Document(
                page_content=self.corpus[index][1],
                metadata={"page_id": self.corpus[index][0], "score": scores[index]},
            )
            for index in order
        ]
        return within_best_ratio(hits, self.min_score_ratio)
