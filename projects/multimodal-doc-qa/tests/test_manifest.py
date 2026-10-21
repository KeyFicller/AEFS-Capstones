"""The manifest says what built the indexes, and load refuses a different answerer of that question."""

import copy
import json
from pathlib import Path

import pytest
import torch
from multimodal_doc_qa.config import Settings
from multimodal_doc_qa.manifest import (
    SCHEMA_VERSION,
    ManifestError,
    fingerprint,
    load_manifest,
    validate,
    write_manifest,
)
from multimodal_doc_qa.schemas import PdfDocument


def _matrices() -> dict[str, torch.Tensor]:
    return {"d/p000": torch.zeros(3, 2), "d/p001": torch.zeros(3, 2)}


def _ocr_docs() -> dict[str, dict]:
    return {"d": {"index": torch.zeros(2, 2), "chunks": [("hello", 0), ("world", 1)]}}


def _write(tmp_path: Path, settings: Settings | None = None, **kwargs: object) -> dict:
    return write_manifest(
        tmp_path,
        settings or Settings(),
        [PdfDocument(doc_id="d")],
        matrices=_matrices(),
        ocr_docs=_ocr_docs(),
        **kwargs,
    )


def test_fingerprint_changes_with_every_input_field() -> None:
    base = {
        "schema_version": SCHEMA_VERSION,
        "render": {"dpi": 180, "long_edge": 2048},
        "embedder": {"model": "a", "dim": 2, "dtype": "float16"},
        "ocr": {"embedder": "b", "dim": 2},
        "chunker": {"max_chars": 400},
    }
    assert fingerprint(base) == fingerprint(copy.deepcopy(base))

    for section, field in (
        ("render", "dpi"),
        ("render", "long_edge"),
        ("embedder", "model"),
        ("embedder", "dim"),
        ("ocr", "embedder"),
        ("chunker", "max_chars"),
    ):
        changed = copy.deepcopy(base)
        changed[section][field] = "different" if isinstance(base[section][field], str) else 1
        assert fingerprint(changed) != fingerprint(base), (section, field)


def test_write_records_counts_and_bytes_per_page(tmp_path: Path) -> None:
    manifest = _write(tmp_path)

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["counts"] == {"documents": 1, "pages": 2, "chunks": 2}
    assert manifest["embedder"]["dim"] == 2
    assert manifest["ocr"]["dim"] == 2
    assert manifest["storage"]["multivector_bytes"] == 2 * 3 * 2 * 4  # pages x patches x dim x 4 bytes
    assert manifest["storage"]["bytes_per_page"] == 24
    assert manifest["fingerprint"].startswith("sha256:")
    assert load_manifest(tmp_path) == manifest


def test_write_hashes_the_source_file(tmp_path: Path) -> None:
    source = tmp_path / "d.pdf"
    source.write_bytes(b"bytes")

    manifest = _write(tmp_path, sources={"d": source})

    assert manifest["documents"][0]["content_sha256"] == (
        "277089d91c0bdf4f2e6862ba7e4a07605119431f5d13f726dd352b06f1b206a9"
    )


def test_validate_accepts_what_write_created(tmp_path: Path) -> None:
    manifest = _write(tmp_path)

    assert validate(tmp_path, Settings(), "visual") == manifest
    assert validate(tmp_path, Settings(), "text") == manifest


def test_validate_rejects_another_embedder(tmp_path: Path) -> None:
    _write(tmp_path)

    with pytest.raises(ManifestError) as exc:
        validate(tmp_path, Settings(embedder_model="other/model"), "visual")

    assert "other/model" in str(exc.value)
    assert "vidore/colSmol-500M" in str(exc.value)


def test_validate_rejects_another_dtype_and_names_both(tmp_path: Path) -> None:
    _write(tmp_path)

    with pytest.raises(ManifestError) as exc:
        validate(tmp_path, Settings(dtype="bfloat16"), "visual")

    assert "bfloat16" in str(exc.value) and "float16" in str(exc.value)


def test_validate_rejects_another_ocr_embedder(tmp_path: Path) -> None:
    _write(tmp_path)

    with pytest.raises(ManifestError):
        validate(tmp_path, Settings(ocr_embedder_model="other/ocr"), "text")


def test_validate_rejects_an_unknown_schema_version(tmp_path: Path) -> None:
    manifest = _write(tmp_path)
    manifest["schema_version"] = 99
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ManifestError):
        validate(tmp_path, Settings(), "visual")


def test_validate_warns_but_returns_none_without_a_manifest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert validate(tmp_path, Settings(), "visual") is None
    assert "manifest" in caplog.text


def test_the_text_kind_ignores_the_vision_embedder(tmp_path: Path) -> None:
    """The BM25 arm reads no vision vectors, so a different vision model must not block it."""
    _write(tmp_path)

    assert validate(tmp_path, Settings(embedder_model="other/model"), "text")
