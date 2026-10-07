"""Retrieval and evidence metrics.

nDCG takes an explicitly ranked page list. The agent loop's page pool is insertion
order, not score order, so a rank metric over it is meaningless. Recall is therefore
measured twice: ``recall_at_k`` over the retriever's own ranking, ``pool_recall`` over
the loop's accumulated pool. ``pool_recall`` above ``recall_at_k`` means the loop
recovered pages the first pass missed.
"""

import math

from multimodal_doc_qa.schemas import Citation


def ndcg_at_k(ranked_page_ids: list[str], relevant: set[str], k: int) -> float:
    """Binary nDCG@k, best first. Returns ``0.0`` when nothing is relevant.

    Duplicate ids are collapsed before ``k`` is applied. The text retriever emits several
    chunks of one page; scoring each chunk would push the ratio above 1.
    """
    ranked = list(dict.fromkeys(ranked_page_ids))[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, pid in enumerate(ranked) if pid in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def bbox_hit_rate(citations: list[Citation], gold: list[Citation]) -> float:
    """Fraction of gold boxes contained by a same-page citation.

    Reference only. Containment (inclusive: a shared edge counts) rewards a box that covers
    the page and misses one that nearly coincides with the gold box. Lead with ``iou_at_threshold``.
    Gold with no bbox is a page-level claim, hit by any citation on that page.
    Returns ``0.0`` when ``gold`` is empty.
    """
    if not gold:
        return 0.0
    hits = 0
    for expected in gold:
        for citation in citations:
            if citation.doc_id != expected.doc_id or citation.page != expected.page:
                continue
            if expected.bbox is None:
                hits += 1
                break
            if citation.bbox is not None and citation.bbox.contains(expected.bbox):
                hits += 1
                break
    return hits / len(gold)


def iou_at_threshold(
    citations: list[Citation], gold: list[Citation], *, threshold: float = 0.5
) -> float:
    """Fraction of gold boxes matched by a same-page citation at IoU >= ``threshold``.

    Primary evidence metric. Record ``threshold`` with the number; without it two runs
    are not comparable. Page-level gold (no bbox) is hit by any citation on that page.
    Returns ``0.0`` when ``gold`` is empty.
    """
    if not 0.0 < threshold <= 1.0:
        # A threshold <= 0 makes a zero-overlap box a hit; > 1 can never match.
        raise ValueError(f"threshold must be in (0.0, 1.0], got {threshold}")
    if not gold:
        return 0.0
    hits = 0
    for expected in gold:
        for citation in citations:
            if citation.doc_id != expected.doc_id or citation.page != expected.page:
                continue
            if expected.bbox is None:
                hits += 1
                break
            if citation.bbox is not None and citation.bbox.iou(expected.bbox) >= threshold:
                hits += 1
                break
    return hits / len(gold)


def recall_at_k(ranked_page_ids: list[str], relevant: set[str], k: int) -> float:
    """Fraction of gold pages the retriever's own ranking found within ``k``.

    Takes an explicitly ranked list: this is the retriever alone, before the agent loop.
    Duplicate ids are collapsed first, the same way ``ndcg_at_k`` does it.
    Returns ``0.0`` when nothing is relevant.
    """
    ranked = list(dict.fromkeys(ranked_page_ids))[:k]
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked)) / len(relevant)


def pool_recall(pool: list[str], relevant: set[str]) -> float:
    """Fraction of gold pages the agent loop's accumulated pool contains.

    The pool is insertion order, not score order, so it is compared as a set.
    Returns ``0.0`` when nothing is relevant.

    ``pool_recall > recall_at_k`` means the loop recovered pages the first pass missed.
    """
    if not relevant:
        return 0.0
    return len(relevant.intersection(pool)) / len(relevant)
