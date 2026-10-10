"""The keyword arm: the LLM extracts the query, BM25 ranks on the result.

No real model is loaded: ``build_chat_model`` is stubbed, and the wrapped retriever records
the query it was handed.
"""

import pytest
from langchain_core.documents import Document
from multimodal_doc_qa.config import Settings
from multimodal_doc_qa.retrievers.keywords import (
    KeywordExtractor,
    KeywordRetriever,
    Keywords,
)


class _InnerStub:
    """A stand-in for the wrapped retriever: records the query it was invoked with."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        return [Document(page_content="chunk", metadata={"page_id": "d/p000", "score": 1.0})]


class _StructuredStub:
    """Stands in for the model bound to ``Keywords``."""

    def __init__(self, keywords: list[str]) -> None:
        self.keywords = keywords
        self.bound: list[object] = []
        self.calls = 0

    def invoke(self, messages: object) -> Keywords:
        self.calls += 1
        return Keywords(keywords=list(self.keywords))


class _ModelStub:
    def __init__(self, keywords: list[str] | None = None) -> None:
        self.structured = _StructuredStub(keywords if keywords is not None else ["margins", "16.8"])

    def with_structured_output(self, schema: object) -> _StructuredStub:
        self.structured.bound.append(schema)
        return self.structured


def _patch_build_chat_model(
    monkeypatch: pytest.MonkeyPatch, model: _ModelStub | None = None
) -> tuple[list[object], _ModelStub]:
    """Patch the host's single construction point; return what it was called with."""
    calls: list[object] = []
    stub = model if model is not None else _ModelStub()

    def build(settings: object) -> _ModelStub:
        calls.append(settings)
        return stub

    monkeypatch.setattr("multimodal_doc_qa.synth.answer.build_chat_model", build)
    return calls, stub


def test_the_inner_retriever_gets_the_keywords_not_the_raw_query() -> None:
    inner = _InnerStub()
    retriever = KeywordRetriever(inner=inner, extractor=lambda query: ["emea", "margin"])

    docs = retriever.invoke("what was the EMEA margin?")

    assert inner.queries == ["emea margin"]
    assert docs[0].metadata["page_id"] == "d/p000"


def test_an_empty_extraction_falls_back_to_the_raw_query() -> None:
    """An empty query scores every document at zero, so it could only return nothing."""
    inner = _InnerStub()
    retriever = KeywordRetriever(inner=inner, extractor=lambda query: [])

    retriever.invoke("what was the EMEA margin?")

    assert inner.queries == ["what was the EMEA margin?"]


def test_constructing_the_extractor_builds_no_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The REPL preloads every mode; this arm must not load a model until it is used."""
    calls, _ = _patch_build_chat_model(monkeypatch)

    extractor = KeywordExtractor(Settings())

    assert calls == []
    extractor("what was the EMEA margin?")
    assert len(calls) == 1
    extractor("and the fees?")
    assert len(calls) == 1  # built once, then reused


def test_the_extractor_binds_the_keywords_schema_and_drops_blanks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank keyword would only add a stray space to the query BM25 tokenizes."""
    model = _ModelStub([" margins ", "", "16.8"])
    _patch_build_chat_model(monkeypatch, model)

    extractor = KeywordExtractor(Settings())
    keywords = extractor("what was the EMEA margin?")
    extractor("and the fees?")

    assert keywords == ["margins", "16.8"]
    assert model.structured.bound == [Keywords]
    assert model.structured.calls == 2
