import hashlib
import random
from pathlib import Path

import pymupdf
import pytest
from multimodal_doc_qa.corpus.generate import _insert_text_layer, generate_corpus
from multimodal_doc_qa.corpus.pages import (
    PAGE_H,
    PAGE_W,
    render_chart_page,
    render_formula_page,
    render_handwriting_page,
    render_paragraph_page,
    render_table_page,
)
from multimodal_doc_qa.schemas import CorpusManifest


def test_paragraph_page_size_and_bboxes_within_bounds() -> None:
    img, drafts, _ = render_paragraph_page(random.Random(0), [("f1", "EMEA margin was 16.8%")])
    assert img.size == (PAGE_W, PAGE_H)
    fact_id, text, bbox = drafts[0]
    assert fact_id == "f1"
    assert text == "EMEA margin was 16.8%"
    assert 0.0 <= bbox.x0 < bbox.x1 <= 1.0
    assert 0.0 <= bbox.y0 < bbox.y1 <= 1.0


def test_table_page_records_every_cell() -> None:
    _, drafts, _ = render_table_page(
        random.Random(0), "Segment Margin", [["EMEA", "16.8%"], ["APAC", "22.1%"]]
    )
    assert {d[1] for d in drafts} >= {"EMEA", "16.8%", "APAC", "22.1%"}


def test_table_page_rejects_no_rows() -> None:
    """``max()`` on an empty list and ``// n_cols`` divide by zero would both raise obscurely."""
    with pytest.raises(ValueError, match="at least one non-empty row"):
        render_table_page(random.Random(0), "Segment Margin", [])
    with pytest.raises(ValueError, match="at least one non-empty row"):
        render_table_page(random.Random(0), "Segment Margin", [[]])


def test_paragraph_page_rejects_more_facts_than_fit_on_the_page() -> None:
    """Past 12 rows the page overflows and the normalized bboxes leave 0..1."""
    with pytest.raises(ValueError, match="at most 12 facts"):
        render_paragraph_page(random.Random(0), [(f"f{i}", f"fact {i}") for i in range(13)])


def test_chart_facts_are_bar_values() -> None:
    img, drafts, _ = render_chart_page(random.Random(1), "Revenue", ["EMEA", "APAC"], [16.8, 22.1])
    assert img.size == (PAGE_W, PAGE_H)
    assert {d[1] for d in drafts} == {"16.8", "22.1"}


def test_chart_text_layer_exposes_labels_but_not_bar_values() -> None:
    """The text layer carries axis labels, never the bar values: that is the point."""
    _, _, runs = render_chart_page(random.Random(1), "Revenue", ["EMEA", "APAC"], [16.8, 22.1])
    text = " ".join(r.text for r in runs)
    assert "EMEA" in text and "APAC" in text
    assert "16.8" not in text and "22.1" not in text


def test_chart_runs_stay_on_page_when_ticks_overshoot() -> None:
    """Matplotlib emits tick labels past the axis limits; none may reach the text layer."""
    _, _, runs = render_chart_page(random.Random(1), "Revenue", ["EMEA", "APAC"], [4.3, 9.7])
    assert runs
    for run in runs:
        assert 0.0 <= run.bbox.x0 < run.bbox.x1 <= 1.0, run
        assert 0.0 <= run.bbox.y0 < run.bbox.y1 <= 1.0, run


def test_formula_and_handwriting_produce_facts() -> None:
    _, fdocs, _ = render_formula_page(random.Random(1), [("f1", r"$E = mc^2$")])
    _, hdocs, _ = render_handwriting_page(random.Random(1), [("h1", "margin 16.8%")])
    assert [d[0] for d in fdocs] == ["f1"]
    assert [d[0] for d in hdocs] == ["h1"]


def test_generate_corpus_is_deterministic_across_output_dirs(tmp_path: Path) -> None:
    m1, q1 = generate_corpus(tmp_path / "a", n_docs=3, seed=7)
    m2, _ = generate_corpus(tmp_path / "b", n_docs=3, seed=7)
    assert m1 == m2, "same seed must produce an identical manifest"
    on_disk = (tmp_path / "a" / "ground_truth.json").read_text()
    assert CorpusManifest.model_validate_json(on_disk) == m1
    assert any(q.hops >= 2 for q in q1), "holdout must contain multi-hop questions"


def test_manifest_paths_are_relative_and_pdfs_exist(tmp_path: Path) -> None:
    manifest, _ = generate_corpus(tmp_path / "c", n_docs=2, seed=1)
    for doc in manifest.docs:
        assert not Path(doc.pdf_path).is_absolute()
        assert (tmp_path / "c" / doc.pdf_path).is_file()


def test_questions_never_leak_their_answer(tmp_path: Path) -> None:
    _, questions = generate_corpus(tmp_path / "d", n_docs=5, seed=2)
    assert questions
    for q in questions:
        assert q.answer.lower() not in q.text.lower(), q.text


def test_multi_hop_evidence_spans_distinct_pages(tmp_path: Path) -> None:
    _, questions = generate_corpus(tmp_path / "e", n_docs=4, seed=3)
    multi = [q for q in questions if q.hops >= 2]
    assert multi
    for q in multi:
        assert len({c.page for c in q.evidence}) >= 2


def test_scanned_pages_are_marked_and_others_are_clean(tmp_path: Path) -> None:
    manifest, _ = generate_corpus(tmp_path / "f", n_docs=5, seed=0)
    assert {p.scanned for doc in manifest.docs for p in doc.pages} == {True, False}


def test_no_two_pages_are_byte_identical(tmp_path: Path) -> None:
    """Identical pages encode identically, so the retriever ties across documents."""
    manifest, _ = generate_corpus(tmp_path / "g", n_docs=5, seed=4)
    seen: set[bytes] = set()
    for doc in manifest.docs:
        pdf = pymupdf.open(tmp_path / "g" / doc.pdf_path)
        for page in doc.pages:
            digest = hashlib.sha256(pdf[page.page].get_pixmap().samples).digest()
            assert digest not in seen, f"duplicate page at {doc.doc_id} p{page.page}"
            seen.add(digest)
        pdf.close()


def test_text_layer_presence_tracks_the_scanned_flag(tmp_path: Path) -> None:
    """Non-scanned pages must be readable without OCR; scanned ones must not be."""
    manifest, _ = generate_corpus(tmp_path / "h", n_docs=5, seed=5)
    for doc in manifest.docs:
        pdf = pymupdf.open(tmp_path / "h" / doc.pdf_path)
        for page in doc.pages:
            text = pdf[page.page].get_text().strip()
            assert (text == "") is page.scanned, f"{doc.doc_id} p{page.page}"


def test_text_layer_round_trips_every_run() -> None:
    """An oversized run spills past the page and PyMuPDF drops it without warning."""
    pages = [
        render_paragraph_page(random.Random(1), [("f1", "EMEA margin was 16.8%")]),
        render_table_page(random.Random(1), "Segment Margin", [["EMEA", "16.8%"]]),
        render_chart_page(random.Random(1), "Revenue", ["EMEA", "APAC"], [4.3, 9.7]),
        render_formula_page(random.Random(1), [("f1", r"$growth = \frac{cur - prev}{prev}$")]),
        render_handwriting_page(random.Random(1), [("h1", "margin 16.8%")]),
    ]
    for _, _, runs in pages:
        pdf = pymupdf.open()
        page = pdf.new_page(width=842.0 * PAGE_W / PAGE_H, height=842.0)
        _insert_text_layer(page, runs)
        extracted = page.get_text()
        for run in runs:
            assert run.text in extracted, run.text
        pdf.close()


def test_pdfs_compress_their_page_images(tmp_path: Path) -> None:
    """Uncompressed RGB pages would make bytes/page meaningless."""
    manifest, _ = generate_corpus(tmp_path / "i", n_docs=2, seed=6)
    for doc in manifest.docs:
        pdf_path = tmp_path / "i" / doc.pdf_path
        raw = len(doc.pages) * PAGE_W * PAGE_H * 3
        assert pdf_path.stat().st_size < raw / 4, f"{doc.doc_id} is not compressed"
