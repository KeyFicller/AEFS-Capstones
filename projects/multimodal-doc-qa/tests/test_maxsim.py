from pathlib import Path

import pytest
import torch
from multimodal_doc_qa.index.maxsim import MultiVectorIndex, maxsim


def _naive_maxsim(query: torch.Tensor, doc: torch.Tensor) -> float:
    """Literal transcription of the definition: per query token its best patch, then sum."""
    return sum(max(float(qi @ dj) for dj in doc) for qi in query)


def _mean_pool_score(query: torch.Tensor, doc: torch.Tensor) -> float:
    """The single-vector alternative, kept here only to show what MaxSim is not."""
    return float((query @ doc.mean(dim=0)).sum())


def test_maxsim_matches_naive_definition() -> None:
    for q, d in (
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([[1.0, 0.0], [0.5, 0.5]])),
        (torch.tensor([[0.3, 0.9]]), torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])),
    ):
        assert abs(float(maxsim(q, d)) - _naive_maxsim(q, d)) < 1e-5


def test_maxsim_rewards_a_page_with_a_patch_for_every_query_token() -> None:
    """Late interaction sees per-token coverage; a pooled vector cannot."""
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    covers_both = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # one patch answers each query token
    repeats_one = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # both patches answer the same token

    assert float(maxsim(query, covers_both)) > float(maxsim(query, repeats_one))
    # pooling is blind to the difference: both pages score the same
    assert _mean_pool_score(query, covers_both) == pytest.approx(
        _mean_pool_score(query, repeats_one)
    )


def test_index_ranks_matching_page_first() -> None:
    idx = MultiVectorIndex(device="cpu")
    idx.add("a", torch.eye(4))
    idx.add("b", torch.tensor([[0.0, 1.0, 0.0, 0.0]]))

    hits = idx.search(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), k=1)

    assert [h.page_id for h in hits] == ["a"]
    assert hits[0].score == pytest.approx(1.0)


def test_save_load_round_trip_preserves_ranking(tmp_path: Path) -> None:
    idx = MultiVectorIndex(device="cpu")
    idx.add("a", torch.eye(4))
    idx.add("b", torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    before = [(h.page_id, h.score) for h in idx.search(query, k=2)]

    path = tmp_path / "nested" / "idx.pt"
    idx.save(path)
    after = [(h.page_id, h.score) for h in MultiVectorIndex.load(path).search(query, k=2)]

    assert after == before


def test_nbytes_counts_each_page_once() -> None:
    """bytes/page is a reported metric, so the accounting must be exact."""
    idx = MultiVectorIndex(device="cpu")
    page_a = torch.eye(4, dtype=torch.float32)  # 4 * 4 * 4 B = 64 B
    page_b = torch.zeros(2, 4, dtype=torch.float32)  # 2 * 4 * 4 B = 32 B
    idx.add("a", page_a)
    idx.add("b", page_b)

    assert idx.nbytes() == 96

    idx.add("a", page_a)  # re-ingesting a page must replace it, not duplicate it
    assert idx.nbytes() == 96


def test_search_clamps_to_available_pages() -> None:
    idx = MultiVectorIndex(device="cpu")
    assert idx.search(torch.eye(4)[0:1], k=5) == []

    idx.add("a", torch.eye(4))
    assert len(idx.search(torch.eye(4)[0:1], k=5)) == 1


def test_a_page_with_no_patches_scores_zero_instead_of_crashing() -> None:
    """One degenerate page must not make ``max(dim=1)`` raise for every query."""
    idx = MultiVectorIndex(device="cpu")
    idx.add("empty", torch.zeros(0, 4))
    idx.add("real", torch.eye(4))

    hits = idx.search(torch.eye(4)[0:1], k=2)

    assert [h.page_id for h in hits] == ["real", "empty"]
    assert hits[1].score == 0.0


def test_a_non_positive_k_returns_nothing() -> None:
    """Python slicing would read ``k=-1`` as "all but the last page"."""
    idx = MultiVectorIndex(device="cpu")
    idx.add("a", torch.eye(4))
    idx.add("b", torch.eye(4))

    assert idx.search(torch.eye(4)[0:1], k=0) == []
    assert idx.search(torch.eye(4)[0:1], k=-1) == []
