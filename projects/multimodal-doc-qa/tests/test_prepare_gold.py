"""Gold conversion: the row filter and the page mapping, with no network."""

from multimodal_doc_qa.eval.prepare_gold import rows_to_questions

_ROW = {
    "doc_id": "watch_d.pdf",
    "question": "How many steps to customize the Down Button?",
    "answer": 2,
    "evidence_pages": "[9, 10]",
}


def test_a_row_becomes_a_page_level_question() -> None:
    (question,) = rows_to_questions([_ROW], {"watch_d.pdf"}, limit=10)

    assert question["text"] == "How many steps to customize the Down Button?"
    assert question["answer"] == "2"
    # evidence_pages is 1-based; the project's page index is 0-based.
    assert question["evidence"] == [
        {"doc_id": "watch_d", "page": 8},
        {"doc_id": "watch_d", "page": 9},
    ]
    assert question["hops"] == 1


def test_rows_outside_the_chosen_docs_are_dropped() -> None:
    assert rows_to_questions([_ROW], {"other.pdf"}, limit=10) == []


def test_unanswerable_rows_are_dropped() -> None:
    row = {**_ROW, "answer": "Not answerable", "evidence_pages": "[]"}
    assert rows_to_questions([row], {"watch_d.pdf"}, limit=10) == []


def test_a_stringified_evidence_pages_is_parsed() -> None:
    """The parquet column is a string like ``"[19, 20]"``, not a list."""
    row = {**_ROW, "evidence_pages": "[19, 20]"}
    (question,) = rows_to_questions([row], {"watch_d.pdf"}, limit=10)
    assert [c["page"] for c in question["evidence"]] == [18, 19]


def test_a_row_with_no_gold_page_is_dropped() -> None:
    """``"[]"`` is a truthy string, so emptiness must be checked after parsing."""
    row = {**_ROW, "evidence_pages": "[]"}
    assert rows_to_questions([row], {"watch_d.pdf"}, limit=10) == []


def test_the_limit_caps_the_number_of_questions() -> None:
    rows = [{**_ROW} for _ in range(5)]
    assert len(rows_to_questions(rows, {"watch_d.pdf"}, limit=2)) == 2


def test_the_limit_is_per_document() -> None:
    """Rows are grouped by document; a global cap would drop whole documents."""
    rows = [{**_ROW, "doc_id": "a.pdf"}, {**_ROW, "doc_id": "b.pdf"}] * 2
    questions = rows_to_questions(rows, {"a.pdf", "b.pdf"}, limit=1)
    assert [q["evidence"][0]["doc_id"] for q in questions] == ["a", "b"]
