"""LangChain retrievers: the seam between a query and an index."""


def within_best_ratio(documents: list, ratio: float) -> list:
    """Keep hits that score at least ``ratio`` times the best hit in this list.

    A ratio of ``0`` keeps every hit. A non-positive best score keeps nothing:
    there is no related page to measure the rest against.
    """
    if not documents or ratio <= 0:
        return documents
    best = max(document.metadata["score"] for document in documents)
    if best <= 0:
        return []
    floor = best * ratio
    return [document for document in documents if document.metadata["score"] >= floor]
