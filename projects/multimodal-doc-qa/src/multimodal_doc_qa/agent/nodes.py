"""Agent nodes: ``plan``, ``assess``, ``verify``. Structured output only; a schema miss is a failure.

``assess`` and ``verify`` are shown page images, not page ids.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from multimodal_doc_qa.schemas import Answer, page_id
from multimodal_doc_qa.synth.answer import build_page_blocks

_PLAN_SYS = (
    "Decompose the question into the minimal independent sub-queries needed to answer it. "
    "Each sub-query must stand alone; do not restate the whole question."
)
_ASSESS_SYS = (
    "You are shown the page images already retrieved for a question. List only the "
    "follow-up sub-queries still needed to answer it; leave the list empty if these pages "
    "are sufficient."
)
_VERIFY_SYS = (
    "You are shown the pages an answer cites. List every claim in the answer that those "
    "pages do not support; leave the list empty if all claims are supported."
)


class Subqueries(BaseModel):
    """``plan`` output: sub-queries that together answer the question."""

    subqueries: list[str] = Field(default_factory=list)


class Followups(BaseModel):
    """``assess`` output: sub-queries still needed; empty means the pool suffices."""

    followups: list[str] = Field(default_factory=list)


class Unsupported(BaseModel):
    """``verify`` output: claims the cited pages do not support; empty means supported."""

    unsupported: list[str] = Field(default_factory=list)


def plan(model: Any, question: str) -> list[str]:
    """Sub-queries for ``question``. Falls back to the question itself when the model returns none."""
    result = model.invoke([
        {"role": "system", "content": _PLAN_SYS},
        {"role": "user", "content": question},
    ])
    return list(result.subqueries) or [question]


def assess(model: Any, question: str, page_ids: list[str], render_dir: Path) -> list[str]:
    """Return the sub-queries still needed, or ``[]`` when the page pool suffices."""
    content = [
        {"type": "text", "text": f"question: {question}"},
        *build_page_blocks(page_ids, render_dir),
    ]
    result = model.invoke([
        {"role": "system", "content": _ASSESS_SYS},
        {"role": "user", "content": content},
    ])
    return list(result.followups)


def verify(
    model: Any, question: str, answer: Answer, pool: list[str], render_dir: Path
) -> list[str]:
    """Claims the citations do not support.

    A citation outside ``pool`` is reported without calling the model: that page was never retrieved.
    """
    known = set(pool)
    cited = list(dict.fromkeys(page_id(c.doc_id, c.page) for c in answer.citations))
    violations = [f"cited page {pid} was never retrieved" for pid in cited if pid not in known]
    inside = [pid for pid in cited if pid in known]
    if not inside:
        return violations
    content = [
        {"type": "text", "text": f"question: {question}\nanswer: {answer.text}\ncited: {', '.join(inside)}"},
        *build_page_blocks(inside, render_dir),
    ]
    result = model.invoke([
        {"role": "system", "content": _VERIFY_SYS},
        {"role": "user", "content": content},
    ])
    return violations + list(result.unsupported)
