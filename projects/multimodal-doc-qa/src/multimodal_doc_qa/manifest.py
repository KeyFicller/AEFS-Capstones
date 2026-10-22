"""What built the artifacts: encoder, dtype, renderer, chunker, and a fingerprint.

Load-time validation is the point. Without it, swapping ``MDQ_EMBEDDER_MODEL`` keeps
loading the old vectors and silently scores noise as if it were recall.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch

from multimodal_doc_qa.config import (
    CHUNK_MAX_CHARS,
    RENDER_DPI,
    RENDER_LONG_EDGE,
    Settings,
    artifact_paths,
    manifest_path,
)
from multimodal_doc_qa.schemas import Document, split_page_id

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
Kind = Literal["visual", "text"]

# Which manifest sections an index actually depends on. The BM25 arm reads no vision
# vectors, so a changed vision model must not block it.
_CHECKS: dict[Kind, tuple[tuple[str, str], ...]] = {
    "visual": (
        ("embedder", "model"),
        ("embedder", "dtype"),
        ("render", "dpi"),
        ("render", "long_edge"),
    ),
    "text": (("ocr", "embedder"), ("chunker", "max_chars")),
}


class ManifestError(Exception):
    """The artifacts were built with different settings than the ones in use."""


def _shape(settings: Settings) -> dict:
    """The settings-derived half of the manifest. Tensors add their dims on top."""
    return {
        "schema_version": SCHEMA_VERSION,
        "render": {"dpi": RENDER_DPI, "long_edge": RENDER_LONG_EDGE},
        "embedder": {"model": settings.embedder_model, "dtype": settings.dtype},
        "ocr": {"embedder": settings.ocr_embedder_model},
        "chunker": {"max_chars": CHUNK_MAX_CHARS},
    }


def fingerprint(shape: dict) -> str:
    """Stable over key order; changes when any fingerprinted value changes."""
    canonical = json.dumps(shape, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content(page_ids: list[str], chunks: int) -> dict:
    """The artifact half of the identity: which pages, and how many OCR chunks.

    The settings alone are not an index identity -- two corpora encoded by the same
    checkpoint would hash the same, and the eval header prints this value.
    """
    return {"pages": sorted(page_ids), "chunks": chunks}


def write_manifest(
    root: Path,
    settings: Settings,
    documents: list[Document],
    *,
    matrices: Mapping[str, torch.Tensor],
    ocr_docs: Mapping[str, dict],
    sources: Mapping[str, Path] | None = None,
) -> dict:
    """Write ``manifest.json`` and return it. Dims and counts come from what is on disk."""
    shape = _shape(settings)
    shape["embedder"]["dim"] = next((int(m.shape[1]) for m in matrices.values()), 0)
    shape["ocr"]["dim"] = next((int(e["index"].shape[1]) for e in ocr_docs.values()), 0)

    pages = len(matrices)
    multivector_bytes = sum(int(v.numel() * v.element_size()) for v in matrices.values())
    ocr_path = artifact_paths(root)[2]
    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **shape,
        "counts": {
            "documents": len(documents),
            "pages": pages,
            "chunks": sum(len(e["chunks"]) for e in ocr_docs.values()),
        },
        "storage": {
            "multivector_bytes": multivector_bytes,
            "ocr_bytes": ocr_path.stat().st_size if ocr_path.is_file() else 0,
            "bytes_per_page": round(multivector_bytes / pages) if pages else 0,
        },
        "documents": [
            {
                "doc_id": doc.doc_id,
                "origin": doc.origin,
                "pages": sum(1 for pid in matrices if split_page_id(pid)[0] == doc.doc_id),
                "content_sha256": (
                    _sha256(sources[doc.doc_id])
                    if sources and doc.doc_id in sources and sources[doc.doc_id].is_file()
                    else None
                ),
            }
            for doc in documents
        ],
    }
    manifest["fingerprint"] = fingerprint(
        {**shape, "content": _content(list(matrices), manifest["counts"]["chunks"])}
    )
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_manifest(root: Path) -> dict | None:
    """The manifest under ``root``, or ``None`` for artifacts ingested before it existed."""
    path = manifest_path(root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"unreadable manifest at {path}: {exc}") from exc


def validate(root: Path, settings: Settings, kind: Kind) -> dict | None:
    """Raise ``ManifestError`` when ``kind`` would be scored by the wrong pipeline.

    A missing manifest is a warning, not an error: artifacts ingested before this feature
    must keep working, and re-running ``ingest`` writes one without re-encoding.
    """
    manifest = load_manifest(root)
    if manifest is None:
        logger.warning(
            "no index manifest at %s; assuming the artifacts match %s",
            manifest_path(root),
            settings.embedder_model,
        )
        return None
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"manifest at {manifest_path(root)} has schema_version "
            f"{manifest.get('schema_version')!r}, expected {SCHEMA_VERSION}; re-run ingest"
        )
    expected = _shape(settings)
    for section, field in _CHECKS[kind]:
        built = manifest.get(section, {}).get(field)
        current = expected[section][field]
        if built != current:
            raise ManifestError(
                f"{kind} index {section}.{field}: artifacts were built with {built!r}, "
                f"settings say {current!r} ({manifest_path(root)}); "
                "re-run ingest or restore the environment"
            )
    return manifest


def status(root: Path, settings: Settings) -> str:
    """``ok`` / ``missing`` / ``mismatch`` for ``/health``. Never raises."""
    try:
        if load_manifest(root) is None:
            return "missing"
        validate(root, settings, "visual")
        validate(root, settings, "text")
    except ManifestError:
        return "mismatch"
    return "ok"
