"""MMDocIR gold conversion: bbox normalization and the 0-based page mapping, no network."""

import json

import pytest

from multimodal_doc_qa.eval.prepare_mmdocir import (
    normalize_bbox,
    pages_per_doc,
    select_docs,
    to_questions,
    write_benchmark,
)

_ANNOTATION = {
    "doc_name": "PH_2016.06.08_Economy-Final.pdf",
    "domain": "Research report / Introduction",
    "page_indices": [18196, 18218],
    "questions": [
        {
            "Q": "How do 5% of Latinos see upward mobility?",
            "A": "Less well-off",
            "page_id": [4],
            "layout_mapping": [
                {"page": 4, "page_size": [612.0, 792.0], "bbox": [366, 229, 514, 383]}
            ],
        }
    ],
}

_EXPECTED_BOX = {"x0": 0.598, "y0": 0.289, "x1": 0.84, "y1": 0.484}


def test_a_bbox_is_normalized_by_the_page_size() -> None:
    assert normalize_bbox([366, 229, 514, 383], [612.0, 792.0]) == _EXPECTED_BOX


def test_a_bbox_outside_the_page_is_clamped() -> None:
    """Five boxes in the real data exceed the page; ``BBox`` would reject them unclamped."""
    assert normalize_bbox([-5, 0, 700, 800], [600.0, 800.0]) == {
        "x0": 0.0,
        "y0": 0.0,
        "x1": 1.0,
        "y1": 1.0,
    }


def test_a_page_with_no_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="page_size"):
        normalize_bbox([0, 0, 1, 1], [0.0, 0.0])


def test_the_page_count_is_inclusive() -> None:
    """``page_indices`` is a closed range: end - start + 1 is 23 for this document."""
    assert pages_per_doc(_ANNOTATION) == 23


def test_page_id_is_already_zero_based() -> None:
    (question,) = to_questions(_ANNOTATION, pdf_pages=23)

    assert question["evidence"] == [
        {"doc_id": "PH_2016.06.08_Economy-Final", "page": 4, "bbox": _EXPECTED_BOX}
    ]


def test_a_pdf_page_count_mismatch_is_fatal() -> None:
    with pytest.raises(ValueError, match="page count"):
        to_questions(_ANNOTATION, pdf_pages=22)


def test_a_gold_page_without_a_box_is_a_page_level_citation() -> None:
    annotation = {
        **_ANNOTATION,
        "questions": [{**_ANNOTATION["questions"][0], "page_id": [4, 7]}],
    }

    (question,) = to_questions(annotation, pdf_pages=23)

    assert [citation.get("bbox") for citation in question["evidence"]] == [_EXPECTED_BOX, None]
    assert question["hops"] == 2


def test_selection_is_round_robin_and_deterministic() -> None:
    rows = [
        {"doc_name": f"{domain}-{index}.pdf", "domain": domain}
        for domain in ("Laws", "News", "Brochure")
        for index in range(3)
    ]

    chosen = select_docs(rows, per_domain=2)

    assert [row["doc_name"] for row in chosen] == [
        "Brochure-0.pdf",
        "Laws-0.pdf",
        "News-0.pdf",
        "Brochure-1.pdf",
        "Laws-1.pdf",
        "News-1.pdf",
    ]
    assert select_docs(list(reversed(rows)), per_domain=2) == chosen


def test_write_benchmark_copies_the_chosen_pdfs_and_writes_questions(tmp_path) -> None:
    import pymupdf

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    document = pymupdf.open()
    for _ in range(23):
        document.new_page()
    document.save(pdf_dir / "PH_2016.06.08_Economy-Final.pdf")

    questions_path, corpus = write_benchmark([_ANNOTATION], pdf_dir, tmp_path / "out", per_domain=1)

    assert (corpus / "PH_2016.06.08_Economy-Final.pdf").is_file()
    assert json.loads(questions_path.read_text())[0]["hops"] == 1


def test_write_benchmark_fails_when_a_pdf_is_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="PH_2016"):
        write_benchmark([_ANNOTATION], tmp_path, tmp_path / "out", per_domain=1)
