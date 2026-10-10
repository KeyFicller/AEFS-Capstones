"""Declarative retrieval arms: components are assembled into a retriever from tables.

Adding an arm is one row in ``ARMS``. ``MODES`` is derived from it, so the CLI and the
console can never disagree about which arms exist or the order Shift-Tab cycles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from multimodal_doc_qa.config import Settings, abstract_paths, artifact_paths
from multimodal_doc_qa.schemas import page_id


class IndexNotFoundError(Exception):
    """The selected retrieval path has no index on disk."""


@dataclass(frozen=True)
class Index:
    """One persisted artifact an arm can depend on."""

    label: str
    path: Callable[[Settings], Path]
    hint: str = " -- run `doc-qa ingest` first"


@dataclass(frozen=True)
class Shared:
    """Encoders built once and reused across arms, so Shift-Tab does not reload a checkpoint."""

    encoder: object = None
    text_embedder: object = None


@dataclass(frozen=True)
class Component:
    """One retrieval stage. ``family`` is what fusion and the status bar key off."""

    index: str
    family: str
    label: str
    build: Callable[[Settings, int, Shared], object]


@dataclass(frozen=True)
class Arm:
    """One ``--mode`` value: the stages it runs and how they are fused.

    ``needs_llm`` marks the arms whose *retrieval* calls a chat model, so the eval run
    header can record the model that was used.
    """

    stages: tuple[str, ...]
    fusion: str | None = None
    needs_llm: bool = False


INDEXES: dict[str, Index] = {
    "multivector": Index("multivector", lambda s: artifact_paths(s.artifacts_dir)[1]),
    "ocr": Index("OCR", lambda s: artifact_paths(s.artifacts_dir)[2]),
    "abstract": Index(
        "abstract",
        lambda s: abstract_paths(s.artifacts_dir)[1],
        hint=" -- re-run ingest with MDQ_ABSTRACTS=1 or MDQ_MODE=abstract",
    ),
}


def _build_maxsim(settings: Settings, k: int, shared: Shared):
    from multimodal_doc_qa.embed.encoder import MultiVectorEncoder
    from multimodal_doc_qa.index.maxsim import MultiVectorIndex
    from multimodal_doc_qa.retrievers.multivector import MultiVectorRetriever

    return MultiVectorRetriever(
        encoder=shared.encoder or MultiVectorEncoder(settings.embedder_model, settings),
        index=MultiVectorIndex.load(INDEXES["multivector"].path(settings)),
        k=k,
        min_score_ratio=settings.min_score_ratio,
    )


def _build_pool(settings: Settings, k: int, shared: Shared):
    from multimodal_doc_qa.embed.encoder import MultiVectorEncoder
    from multimodal_doc_qa.index.maxsim import MultiVectorIndex
    from multimodal_doc_qa.retrievers.pool import PooledRetriever, mean_vector

    index = MultiVectorIndex.load(INDEXES["multivector"].path(settings))
    return PooledRetriever(
        encoder=shared.encoder or MultiVectorEncoder(settings.embedder_model, settings),
        pages={page_id: mean_vector(matrix) for page_id, matrix in index.matrices().items()},
        k=k,
        min_score_ratio=settings.min_score_ratio,
    )


def _build_text(settings: Settings, k: int, shared: Shared, index_name: str):
    import torch

    from multimodal_doc_qa.retrievers.text import MultiDocTextRetriever, OcrEmbedder, TextRetriever

    embedder = shared.text_embedder or OcrEmbedder(settings.ocr_embedder_model, settings.device)
    payload = torch.load(INDEXES[index_name].path(settings), weights_only=True)
    return MultiDocTextRetriever(
        retrievers=[
            TextRetriever(
                doc_id=doc_id, embedder=embedder, index=entry["index"], chunks=entry["chunks"], k=k
            )
            for doc_id, entry in payload.items()
        ],
        k=k,
        min_score_ratio=settings.min_score_ratio,
    )


def _build_bm25(settings: Settings, k: int, shared: Shared):
    import torch

    from multimodal_doc_qa.retrievers.bm25 import BM25Retriever

    payload = torch.load(INDEXES["ocr"].path(settings), weights_only=True)
    corpus = [
        (page_id(doc_id, int(page)), text)
        for doc_id, entry in payload.items()
        for text, page in entry["chunks"]
    ]
    return BM25Retriever(corpus=corpus, k=k, min_score_ratio=settings.min_score_ratio)


def _build_bm25_kw(settings: Settings, k: int, shared: Shared):
    """The same BM25, but the query is the LLM's keywords. The model is built on first use."""
    from multimodal_doc_qa.retrievers.keywords import KeywordExtractor, KeywordRetriever

    return KeywordRetriever(
        inner=_build_bm25(settings, k, shared),
        extractor=KeywordExtractor(settings),
    )


COMPONENTS: dict[str, Component] = {
    "maxsim": Component("multivector", "embedding", "{embedder}", _build_maxsim),
    "pool": Component("multivector", "embedding", "{embedder}", _build_pool),
    "ocr": Component("ocr", "embedding", "{ocr}", lambda s, k, sh: _build_text(s, k, sh, "ocr")),
    "abstract": Component(
        "abstract", "embedding", "{describer}  {ocr}", lambda s, k, sh: _build_text(s, k, sh, "abstract")
    ),
    "bm25": Component("ocr", "keywords", "bm25", _build_bm25),
    "bm25_kw": Component("ocr", "keywords", "{extractor}  bm25", _build_bm25_kw),
}

ARMS: dict[str, Arm] = {
    "maxsim": Arm(("maxsim",)),
    "pool": Arm(("pool",)),
    "ocr": Arm(("ocr",)),
    "abstract": Arm(("abstract",)),
    "lexical": Arm(("bm25",)),
    "lexical-kw": Arm(("bm25_kw",), needs_llm=True),
    "hybrid-ocr": Arm(("ocr", "bm25"), fusion="rrf"),
    "hybrid-abstract": Arm(("abstract", "bm25"), fusion="rrf"),
    "hybrid-pool": Arm(("pool", "bm25"), fusion="rrf"),
    "hybrid-maxsim": Arm(("maxsim", "bm25"), fusion="rrf"),
}

MODES: tuple[str, ...] = tuple(ARMS)


def require_index(name: str, settings: Settings) -> None:
    """Raise ``IndexNotFoundError`` when the artifact an arm needs is not on disk."""
    spec = INDEXES[name]
    path = spec.path(settings)
    if not path.is_file():
        raise IndexNotFoundError(f"no {spec.label} index at {path}{spec.hint}")


def _fuse_rrf(parts: dict[str, object], k: int):
    from multimodal_doc_qa.retrievers.hybrid import HybridRetriever

    by_family = {COMPONENTS[stage].family: part for stage, part in parts.items()}
    return HybridRetriever(dense=by_family["embedding"], lexical=by_family["keywords"], k=k)


FUSIONS: dict[str, Callable[[dict[str, object], int], object]] = {"rrf": _fuse_rrf}


def build_retriever(
    settings: Settings,
    mode: str,
    *,
    encoder: object = None,
    text_embedder: object = None,
    k: int | None = None,
):
    """Assemble the retriever for ``mode``. ``KeyError`` for an unknown mode, ``IndexNotFoundError`` for a missing index."""
    arm = ARMS[mode]
    top_k = k if k is not None else settings.top_k
    for name in {COMPONENTS[stage].index for stage in arm.stages}:
        require_index(name, settings)
    shared = Shared(encoder=encoder, text_embedder=text_embedder)
    stage_k = 4 * top_k if arm.fusion else top_k
    parts = {stage: COMPONENTS[stage].build(settings, stage_k, shared) for stage in arm.stages}
    if arm.fusion is None:
        return next(iter(parts.values()))
    return FUSIONS[arm.fusion](parts, k=top_k)
