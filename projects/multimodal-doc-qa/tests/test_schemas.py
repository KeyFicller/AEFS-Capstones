import pytest
from multimodal_doc_qa.schemas import (
    BBox,
    Citation,
    ImageDocument,
    PdfDocument,
    Question,
    TextDocument,
    load_documents,
    save_documents,
)
from pydantic import ValidationError


def test_bbox_contains_normalized_inner() -> None:
    outer = BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.9)
    assert outer.contains(BBox(x0=0.2, y0=0.2, x1=0.3, y1=0.3))
    assert not outer.contains(BBox(x0=0.0, y0=0.2, x1=0.3, y1=0.3))


def test_bbox_rejects_an_inverted_rectangle() -> None:
    """Inverted corners are not a rectangle: ``contains`` would answer False for everything."""
    with pytest.raises(ValidationError):
        BBox(x0=0.5, y0=0.5, x1=0.1, y1=0.1)


def test_bbox_rejects_a_partially_inverted_rectangle() -> None:
    with pytest.raises(ValidationError):
        BBox(x0=0.1, y0=0.9, x1=0.3, y1=0.2)


def test_bbox_rejects_coordinates_outside_the_normalized_range() -> None:
    with pytest.raises(ValidationError):
        BBox(x0=-0.1, y0=0.0, x1=0.5, y1=0.5)


def test_bbox_accepts_a_degenerate_but_ordered_rectangle() -> None:
    """A zero-area box is still a location, and the layout can legitimately produce one."""
    assert BBox(x0=0.2, y0=0.3, x1=0.2, y1=0.3)


def test_document_catalog_roundtrips_the_three_origins(tmp_path) -> None:
    docs = [
        ImageDocument(doc_id="shot"),
        PdfDocument(doc_id="doc000"),
        TextDocument(doc_id="note", pages=["The tanh gate starts at zero."]),
    ]
    path = tmp_path / "documents.json"

    save_documents(docs, path)

    loaded = load_documents(path)
    assert isinstance(loaded["shot"], ImageDocument)
    assert isinstance(loaded["doc000"], PdfDocument)
    assert loaded["note"] == docs[2]


def test_question_roundtrips_through_json() -> None:
    q = Question(
        qid="q1",
        text="What was EMEA margin?",
        answer="16.8%",
        evidence=[Citation(doc_id="d1", page=2, bbox=None)],
        hops=2,
    )
    assert Question.model_validate_json(q.model_dump_json()) == q


def test_a_negative_page_is_rejected() -> None:
    """A negative page is a valid Python index that silently reads the wrong page."""
    with pytest.raises(ValidationError):
        Citation(doc_id="doc000", page=-1)


def test_a_catalog_with_a_duplicate_doc_id_is_rejected(tmp_path) -> None:
    """``doc_id`` is the lookup key; collapsing a duplicate would drop a document."""
    path = tmp_path / "documents.json"
    save_documents([PdfDocument(doc_id="dup"), PdfDocument(doc_id="dup")], path)

    with pytest.raises(ValueError, match="duplicate doc_id"):
        load_documents(path)
