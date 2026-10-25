"""Lexical arm: BM25 over the OCR chunks. Hand-written so the ablation adds no dependency."""

from __future__ import annotations

import heapq
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
    """Corpus statistics for BM25, plus the inverted index a query needs to skip chunks."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        """Precompute document frequency, lengths, the mean length and the inverted index."""
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n = len(corpus)
        self.lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.lengths) / self.n if self.n else 0.0
        self.tf = [Counter(doc) for doc in corpus]
        self.df = Counter(term for doc in corpus for term in set(doc))
        # term -> the chunks holding it. Built here, at startup, not per query: it lets a
        # query skip every chunk no term in it appears in.
        self.postings: dict[str, list[int]] = {}
        for index, doc in enumerate(corpus):
            for term in set(doc):
                self.postings.setdefault(term, []).append(index)

    def idf(self, term: str) -> float:
        """Inverse document frequency of ``term``, never negative."""
        df = self.df[term]
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

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
            denominator = frequency + self.k1 * (1.0 - self.b + self.b * length / self.avgdl)
            total += self.idf(term) * frequency * (self.k1 + 1.0) / denominator
        return total

    def top_k(self, query: list[str], k: int) -> list[int]:
        """Indices of the best ``k`` chunks, best first. Exact, not approximate.

        Candidates come from the inverted index. Each is scored in descending order of an
        upper bound -- a term adds at most ``idf * (k1 + 1)``, since its length
        normalisation divides by at least the term frequency -- so a candidate bounded
        below the kth score, and every one behind it, cannot enter the result.
        """
        if k <= 0 or self.avgdl == 0.0:
            return []
        bound: dict[int, float] = {}
        for term in set(query):
            docs = self.postings.get(term)
            if not docs:
                continue
            ceiling = self.idf(term) * (self.k1 + 1.0)
            for doc in docs:
                bound[doc] = bound.get(doc, 0.0) + ceiling

        # ``-index`` makes the heap drop the higher index on a tie, so equal scores keep
        # the ascending-index order a stable sort of the whole corpus would give.
        best: list[tuple[float, int]] = []
        for doc in sorted(bound, key=bound.__getitem__, reverse=True):
            if len(best) == k and bound[doc] < best[0][0]:
                break
            scored = (self.score(query, doc), -doc)
            if len(best) < k:
                heapq.heappush(best, scored)
            elif scored > best[0]:
                heapq.heapreplace(best, scored)
        return [-negated for _, negated in sorted(best, key=lambda pair: (-pair[0], -pair[1]))]


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
        order = self._bm25.top_k(tokens, self.k)
        hits = [
            Document(
                page_content=self.corpus[index][1],
                metadata={
                    "page_id": self.corpus[index][0],
                    "score": self._bm25.score(tokens, index),
                    "snippet": self.corpus[index][1],
                },
            )
            for index in order
        ]
        return within_best_ratio(hits, self.min_score_ratio)
