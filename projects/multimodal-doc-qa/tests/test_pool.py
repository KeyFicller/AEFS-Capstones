import torch
from langchain_core.documents import Document
from multimodal_doc_qa.index.maxsim import MultiVectorIndex
from multimodal_doc_qa.retrievers.pool import PooledRetriever, mean_vector


def test_mean_vector_averages_rows_and_normalizes() -> None:
    pooled = mean_vector(torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    assert torch.allclose(pooled, torch.tensor([1.0, 0.0]))


def test_pooled_retrieval_ranks_the_page_whose_mean_matches_the_query() -> None:
    """A page of patches along x beats a page of patches along y for an x-aligned query."""
    index = MultiVectorIndex()
    index.add("a/p000", torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    index.add("b/p000", torch.tensor([[0.0, 1.0], [0.0, 1.0]]))

    class _Encoder:
        def encode_query(self, text: str) -> torch.Tensor:
            return torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    hits = PooledRetriever(
        encoder=_Encoder(),
        pages={page_id: mean_vector(matrix) for page_id, matrix in index.matrices().items()},
        k=2,
        min_score_ratio=0.5,
    ).invoke("along x")

    assert [hit.metadata["page_id"] for hit in hits] == ["a/p000"]


def test_a_zero_mean_stays_zero() -> None:
    pooled = mean_vector(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    assert torch.allclose(pooled, torch.zeros(2))
    assert pooled.shape == (2,)
