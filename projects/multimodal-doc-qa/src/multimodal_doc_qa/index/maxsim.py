"""Exact MaxSim over page patch matrices.

Each query token takes its best page patch; those maxima are summed. A single
matching patch cannot carry the page.
"""

from pathlib import Path

import torch

from multimodal_doc_qa.schemas import ScoredPage


def maxsim(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """``sum_i max_j query[i] · doc[j]``. ``query`` is ``[tokens, dim]``, ``doc`` is ``[patches, dim]``."""

    sims = query @ doc.T
    return sims.max(dim=1).values.sum()


class MultiVectorIndex:
    """In-memory patch matrices and exact MaxSim. Brute force is enough for a few hundred pages."""

    def __init__(self, device: str = "cpu") -> None:
        """Create an empty index; search runs on ``device``."""
        self.device = torch.device(device)
        self._vectors: dict[str, torch.Tensor] = {}

    def add(self, page_id: str, vectors: torch.Tensor) -> None:
        """Store the ``[n_patches, dim]`` matrix for ``page_id``, replacing any previous one."""
        self._vectors[page_id] = vectors.to(self.device)

    def search(self, query: torch.Tensor, k: int) -> list[ScoredPage]:
        """Return the ``k`` highest-scoring pages, best first.

        Returns fewer than ``k`` when the index holds fewer pages, and ``[]`` when
        it is empty.
        """
        q = query.to(self.device)
        scored = [
            ScoredPage(page_id=page_id, score=float(maxsim(q, v)))
            for page_id, v in self._vectors.items()
        ]
        return sorted(scored, key=lambda page: page.score, reverse=True)[:k]

    def save(self, path: Path) -> None:
        """Persist the index to ``path`` (parent directories are created)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {pid: v.cpu() for pid, v in self._vectors.items()},
            path
        )

    @classmethod
    def load(cls, path: Path) -> MultiVectorIndex:
        """Load an index written by ``save``."""
        obj = cls(device="cpu")
        obj._vectors = {
            pid: v for pid, v in torch.load(path, weights_only=True).items()
        }
        return obj

    def nbytes(self) -> int:
        """Bytes held by the stored vectors -- the ``bytes/page`` metric denominator."""
        return sum(v.numel() * v.element_size() for v in self._vectors.values())
