import io
import json
import random
from pathlib import Path
from typing import NamedTuple

import pymupdf
from PIL import Image

from multimodal_doc_qa.corpus.pages import (
    PAGE_H,
    PAGE_W,
    FactDraft,
    TextRun,
    render_chart_page,
    render_formula_page,
    render_handwriting_page,
    render_paragraph_page,
    render_table_page,
)
from multimodal_doc_qa.schemas import (
    Citation,
    CorpusManifest,
    DocSpec,
    Fact,
    PageKind,
    PageSpec,
    Question,
)

PAGES_PER_DOC = 3

_KINDS: tuple[PageKind, ...] = ("paragraph", "table", "chart", "formula", "handwriting")

# Only these page kinds are worth degrading into scans: they are where the OCR
# baseline struggles even on clean input.
_SCANNABLE: frozenset[PageKind] = frozenset({"handwriting", "chart"})
_SCAN_BACKGROUND = 235

# Invisible text layer sizing. Helvetica advances roughly 0.55 em per glyph; the
# text is drawn from the run origin rightwards, so anything wider would push glyphs
# past the page edge, where get_text() drops them without warning.
_ADVANCE_RATIO = 0.55
_TEXT_LAYER_MAX_SIZE = 24.0

# Page content is keyed by a global page index so no two pages share a figure.
# Byte-identical pages encode to identical embeddings, which lets the retriever tie
# across documents and compresses the nDCG gap the benchmark exists to measure.
_SEGMENTS: tuple[str, ...] = ("EMEA", "APAC", "North America", "LatAm", "Nordics", "MEA")
_FIGURES: tuple[float, ...] = (
    16.8, 22.1, 4.3, 9.7, 31.4, 12.6, 18.9, 7.2, 25.6, 11.3,
    29.8, 6.5, 20.4, 13.7, 8.1, 27.3, 15.2, 33.9, 10.4, 24.7,
)
_FORMULAS: tuple[str, ...] = (
    r"$margin = \frac{rev - cost}{rev}$",
    r"$yield = \frac{gross}{net} \times 100$",
    r"$growth = \frac{cur - prev}{prev}$",
    r"$cost = fixed + variable$",
    r"$roi = \frac{gain - cost}{cost}$",
)

_TABLE_TITLE = "Segment Margin"
_CHART_TITLE = "Revenue by Region"


def _figure_at(n: int) -> float:
    return _FIGURES[n % len(_FIGURES)]


def _segment_at(n: int) -> str:
    return _SEGMENTS[n % len(_SEGMENTS)]


class _BuiltPage(NamedTuple):
    image: Image.Image
    facts: list[FactDraft]
    subjects: dict[str, str]  # fact_id -> what a question asks about
    text_runs: list[TextRun]


def _build_page(rng: random.Random, kind: PageKind, n: int) -> _BuiltPage:
    """Render page ``n`` and map each answerable fact to the subject it is asked about.

    Only facts with a subject become question targets; the rest stay in the
    manifest as ground truth (e.g. table headers and row labels).
    """
    if kind == "paragraph":
        subject = _segment_at(n)
        sentence = f"{subject} margin was {_figure_at(n)}%"
        img, drafts, runs = render_paragraph_page(rng, [("f0", sentence)])
        return _BuiltPage(img, drafts, {"f0": subject}, runs)
    if kind == "table":
        rows = [
            ["Segment", "Margin"],
            [_segment_at(n), f"{_figure_at(n)}%"],
            [_segment_at(n + 1), f"{_figure_at(n + 1)}%"],
        ]
        img, drafts, runs = render_table_page(rng, _TABLE_TITLE, rows)
        # Skip the header row and label column: their value *is* their label.
        subjects = {
            f"cell_{ri}_{ci}": f"{rows[ri][0]} {rows[0][ci]}"
            for ri in range(1, len(rows))
            for ci in range(1, len(rows[0]))
        }
        return _BuiltPage(img, drafts, subjects, runs)
    if kind == "chart":
        labels = [_segment_at(n), _segment_at(n + 1)]
        values = [_figure_at(n), _figure_at(n + 1)]
        img, drafts, runs = render_chart_page(rng, _CHART_TITLE, labels, values)
        subjects = {f"bar_{i}": f"{label} revenue" for i, label in enumerate(labels)}
        return _BuiltPage(img, drafts, subjects, runs)
    if kind == "formula":
        expr = _FORMULAS[n % len(_FORMULAS)]
        img, drafts, runs = render_formula_page(rng, [("f0", expr)])
        return _BuiltPage(img, drafts, {"f0": "the margin formula"}, runs)

    subject = _segment_at(n)
    sentence = f"{subject} margin was {_figure_at(n)}%"
    img, drafts, runs = render_handwriting_page(rng, [("f0", sentence)])
    return _BuiltPage(img, drafts, {"f0": subject}, runs)


def _insert_text_layer(pdf_page: pymupdf.Page, runs: list[TextRun]) -> None:
    """Add an invisible text layer mirroring the page's visible text.

    This is what a digital-born page carries natively and what the OCR baseline
    reads without OCR. Scanned pages deliberately get none.
    """
    rect = pdf_page.rect
    for run in runs:
        ink_h = (run.bbox.y1 - run.bbox.y0) * rect.height
        ink_w = (run.bbox.x1 - run.bbox.x0) * rect.width
        # PyMuPDF draws from the run origin rightwards, and get_text() silently drops
        # glyphs that spill past the page. Cap the size so the string fits its run.
        size = min(ink_h, _TEXT_LAYER_MAX_SIZE, ink_w / (_ADVANCE_RATIO * len(run.text)))
        pdf_page.insert_text(
            (run.bbox.x0 * rect.width, run.bbox.y1 * rect.height),
            run.text,
            fontsize=max(1.0, size),
            render_mode=3,  # invisible: extractable but never painted
        )


def generate_corpus(
    out_dir: Path, n_docs: int, seed: int
) -> tuple[CorpusManifest, list[Question]]:
    """Write a reproducible synthetic corpus and its holdout questions.

    Writes ``<doc_id>.pdf``, ``ground_truth.json`` and ``questions.json`` into
    ``out_dir`` and returns the manifest alongside the questions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    docs: list[DocSpec] = []
    subjects_by_page: dict[tuple[str, int], dict[str, str]] = {}

    for d in range(n_docs):
        doc_id = f"doc{d:03d}"
        pdf = pymupdf.open()
        pages: list[PageSpec] = []
        for p in range(PAGES_PER_DOC):
            kind = _KINDS[(d + p) % len(_KINDS)]
            built = _build_page(rng, kind, d * PAGES_PER_DOC + p)

            scanned = p == PAGES_PER_DOC - 1 and kind in _SCANNABLE
            img = built.image
            if scanned:
                img = Image.eval(img, lambda v: _SCAN_BACKGROUND if v > 200 else v)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            # Fix the page height at A4 and derive the width from the canvas ratio,
            # so the encoder never sees a distorted page.
            page_h = 842.0
            pdf_page = pdf.new_page(width=page_h * PAGE_W / PAGE_H, height=page_h)
            pdf_page.insert_image(pdf_page.rect, stream=buf.getvalue())
            if not scanned:
                _insert_text_layer(pdf_page, built.text_runs)

            pages.append(
                PageSpec(
                    doc_id=doc_id,
                    page=p,
                    kind=kind,
                    scanned=scanned,
                    facts=[
                        Fact(fact_id=fid, text=text, page=p, bbox=bbox)
                        for fid, text, bbox in built.facts
                    ],
                )
            )
            subjects_by_page[(doc_id, p)] = built.subjects

        # Store the path relative to the corpus root: an absolute path would make
        # the manifest machine-specific and break same-seed reproducibility.
        pdf_name = f"{doc_id}.pdf"
        # Default save() embeds page images as uncompressed RGB (6.5 MB/page at
        # this canvas size), so force Flate on both streams and images.
        pdf.save(out_dir / pdf_name, deflate=True, deflate_images=True)
        pdf.close()
        docs.append(DocSpec(doc_id=doc_id, pdf_path=pdf_name, pages=pages))

    manifest = CorpusManifest(seed=seed, docs=docs)
    (out_dir / "ground_truth.json").write_text(manifest.model_dump_json(indent=2))

    questions = _build_questions(docs, subjects_by_page)
    (out_dir / "questions.json").write_text(
        json.dumps([q.model_dump() for q in questions], indent=2)
    )
    return manifest, questions


def _citation(page: PageSpec, fact: Fact) -> Citation:
    return Citation(doc_id=page.doc_id, page=page.page, bbox=fact.bbox)


def _build_questions(
    docs: list[DocSpec], subjects_by_page: dict[tuple[str, int], dict[str, str]]
) -> list[Question]:
    """Build holdout questions from fact subjects, never from the answer text."""
    questions: list[Question] = []

    for doc in docs:
        picked: list[tuple[PageSpec, Fact, str]] = []
        for page in doc.pages:
            subjects = subjects_by_page[(doc.doc_id, page.page)]
            for fact in page.facts:
                subject = subjects.get(fact.fact_id)
                if subject is not None:
                    picked.append((page, fact, subject))
                    break

        for page, fact, subject in picked:
            questions.append(
                Question(
                    qid=f"{doc.doc_id}-p{page.page}",
                    text=f"In {doc.doc_id} page {page.page}, what is stated about {subject}?",
                    answer=fact.text,
                    evidence=[_citation(page, fact)],
                    hops=1,
                )
            )

        if len(picked) >= 2:
            (page_a, fact_a, subj_a), (page_b, fact_b, subj_b) = picked[0], picked[1]
            questions.append(
                Question(
                    qid=f"{doc.doc_id}-h2",
                    text=(
                        f"In {doc.doc_id}, compare what page {page_a.page} states about "
                        f"{subj_a} with what page {page_b.page} states about {subj_b}."
                    ),
                    answer=f"{fact_a.text} | {fact_b.text}",
                    evidence=[_citation(page_a, fact_a), _citation(page_b, fact_b)],
                    hops=2,
                )
            )

    return questions


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "_artifacts"
    manifest, questions = generate_corpus(out_dir, n_docs=2, seed=42)

    for doc in manifest.docs:
        pages = ", ".join(
            f"p{page.page}:{page.kind}{' (scan)' if page.scanned else ''}"
            for page in doc.pages
        )
        print(f"{doc.doc_id} -> {doc.pdf_path}  [{pages}]")
    print(f"{len(questions)} questions -> {out_dir / 'questions.json'}")
