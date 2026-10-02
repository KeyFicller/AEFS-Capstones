"""Corpus and QA schemas. Page ids are ``{doc_id}/p{page:03d}``."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

PageKind = Literal["paragraph", "table", "chart", "handwriting", "formula"]


def page_id(doc_id: str, page: int) -> str:
    """Canonical page id, e.g. ``doc000/p002``. Build every id through here.

    A bare ``p002`` collides across documents and does not match the rendered filename.
    """
    return f"{doc_id}/p{page:03d}"


class BBox(BaseModel):
    """Leaf of the corpus tree: normalized (0..1) rectangle."""
    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _corners_are_ordered(self) -> BBox:
        """Reject inverted corners. ``ge``/``le`` alone accept ``x0 > x1``, which then contains nothing."""
        if self.x0 > self.x1 or self.y0 > self.y1:
            raise ValueError(
                f"bbox corners are inverted: x0={self.x0} x1={self.x1} "
                f"y0={self.y0} y1={self.y1}"
            )
        return self

    def contains(self, inner: BBox) -> bool:
        return (
            self.x0 <= inner.x0
            and self.x1 >= inner.x1
            and self.y0 <= inner.y0
            and self.y1 >= inner.y1
        )

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def iou(self, other: BBox) -> float:
        """Intersection over union, or ``0.0`` when the boxes do not overlap."""
        x0, y0 = max(self.x0, other.x0), max(self.y0, other.y0)
        x1, y1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        intersection = (x1 - x0) * (y1 - y0)
        return intersection / (self.area + other.area - intersection)

class Citation(BaseModel):
    """One citation in an answer: doc_id + page always present; bbox may be absent."""
    doc_id: str
    page: int
    bbox: BBox | None = None

class Answer(BaseModel):
    """Answerer output: text plus citations (mirrors Question.evidence)."""
    text: str
    citations: list[Citation] = Field(default_factory=list)

class ScoredPage(BaseModel):
    """Retrieval output; a standalone leaf consumed only by retrievers and eval."""
    page_id: str
    score: float

class Fact(BaseModel):
    """One answerable fact on a page, carrying its own bbox: the ground-truth unit."""
    fact_id: str
    text: str
    page: int
    bbox: BBox

class PageSpec(BaseModel):
    """One page: kind selects the renderer, scanned decides whether a text layer exists."""
    doc_id: str
    page: int
    kind: PageKind
    scanned: bool
    facts: list[Fact] = Field(default_factory=list)

class DocSpec(BaseModel):
    """One PDF document plus all of its pages."""
    doc_id: str
    pdf_path: str
    pages: list[PageSpec]

class CorpusManifest(BaseModel):
    """Corpus tree root: generation seed plus every document (seed reproduces the corpus)."""
    seed: int
    docs: list[DocSpec]

class Question(BaseModel):
    """Holdout question: evidence marks the answer's page / box, hops how many pages it needs."""
    qid: str
    text: str
    answer: str
    evidence: list[Citation]
    hops: int = 1

