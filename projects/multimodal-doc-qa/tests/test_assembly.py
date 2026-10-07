"""The arm tables are the single source of truth: MODES, dispatch, and missing-index errors."""

from pathlib import Path

import pytest

from multimodal_doc_qa.config import Settings
from multimodal_doc_qa.retrievers import assembly
from multimodal_doc_qa.retrievers.assembly import (
    ARMS,
    COMPONENTS,
    INDEXES,
    MODES,
    IndexNotFoundError,
    build_retriever,
)

ORDER = (
    "maxsim",
    "pool",
    "ocr",
    "abstract",
    "lexical",
    "hybrid-ocr",
    "hybrid-abstract",
    "hybrid-pool",
    "hybrid-maxsim",
)


def test_modes_is_the_arm_table_in_order() -> None:
    assert MODES == ORDER
    assert tuple(ARMS) == ORDER


def test_every_stage_names_a_component_and_an_index() -> None:
    for arm in ARMS.values():
        for stage in arm.stages:
            assert stage in COMPONENTS
            assert COMPONENTS[stage].index in INDEXES


def test_every_component_family_is_known() -> None:
    for component in COMPONENTS.values():
        assert component.family in {"embedding", "keywords"}


def test_the_hybrid_arms_pair_one_embedding_with_one_keywords() -> None:
    for name, arm in ARMS.items():
        if arm.fusion is None:
            continue
        families = sorted(COMPONENTS[stage].family for stage in arm.stages)
        assert families == ["embedding", "keywords"], name


@pytest.fixture
def depths(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the ``k`` every component is built with; stub out index checks and the builders."""
    seen: list[int] = []
    monkeypatch.setattr(assembly, "require_index", lambda name, settings: None)
    for name, component in list(COMPONENTS.items()):
        monkeypatch.setitem(
            COMPONENTS,
            name,
            assembly.Component(
                component.index,
                component.family,
                component.label,
                lambda settings, k, shared: seen.append(k) or object(),
            ),
        )
    return seen


def test_a_flat_arm_keeps_top_k(depths: list[int]) -> None:
    build_retriever(Settings(top_k=5), "maxsim")

    assert depths == [5]


def test_a_fused_arm_queries_both_stages_deeper(depths: list[int]) -> None:
    build_retriever(Settings(top_k=5), "hybrid-ocr")

    assert depths == [20, 20]


def test_a_missing_index_names_the_index(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFoundError) as exc:
        build_retriever(Settings(artifacts_dir=tmp_path), "lexical")

    assert "OCR" in str(exc.value)


def test_an_unknown_mode_is_a_key_error() -> None:
    with pytest.raises(KeyError):
        build_retriever(Settings(), "telepathy")
