"""Turn an MMDocIR subset into this project's corpus directory and questions.json.

Run: ``HF_ENDPOINT=https://hf-mirror.com python -m multimodal_doc_qa.eval.prepare_mmdocir``.

Two facts the mapping depends on:

- ``page_id`` is **0-based**, already the PDF page index. The README reports 20,395 page
  screenshots, which is ``sum(page_indices[1] - page_indices[0] + 1)`` -- so ``page_indices``
  is a closed range and no page offset is applied here. Do not copy the ``- 1`` from
  ``prepare_gold.py``: MMLongBench-Doc is 1-based, this set is not.
- ``layout_mapping`` carries the box gold: ``bbox`` is in points of ``page_size``. It is
  normalized to 0..1 and clamped, because a handful of boxes run past the page edge and
  ``BBox`` rejects anything outside the unit square.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from multimodal_doc_qa.config import PROJECT_ROOT

DATASET = "MMDocIR/MMDocIR_Evaluation_Dataset"
ANNOTATIONS = "MMDocIR_annotations.jsonl"
PDF_ARCHIVE = "doc_miscellaneous/doc_pdfs.rar"
ARCHIVE_PREFIX = "doc_pdfs"


def normalize_bbox(bbox: list[float], page_size: list[float]) -> dict[str, float]:
    """``[x1, y1, x2, y2]`` in points -> normalized corners, clamped to the page."""
    width, height = (float(value) for value in page_size)
    if width <= 0 or height <= 0:
        raise ValueError(f"page_size must be positive, got {page_size!r}")
    left, top, right, bottom = (float(value) for value in bbox)
    corners = (left / width, top / height, right / width, bottom / height)
    x0, y0, x1, y1 = (round(min(1.0, max(0.0, value)), 3) for value in corners)
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def pages_per_doc(annotation: dict) -> int:
    """Page count from the annotation's closed row range."""
    start, end = annotation["page_indices"]
    return int(end) - int(start) + 1


def select_docs(annotations: list[dict], per_domain: int) -> list[dict]:
    """``per_domain`` documents from every domain, round-robin, deterministic.

    Round-robin rather than "first N overall": the rows are grouped by domain, so a global
    cap would take a handful of domains and drop the rest. Within a domain the order is
    ``doc_name``, so the same subset comes out of any run.
    """
    by_domain: dict[str, list[dict]] = {}
    for annotation in annotations:
        by_domain.setdefault(annotation["domain"], []).append(annotation)
    for group in by_domain.values():
        group.sort(key=lambda annotation: annotation["doc_name"])
    return [
        by_domain[domain][rank]
        for rank in range(per_domain)
        for domain in sorted(by_domain)
        if rank < len(by_domain[domain])
    ]


def to_questions(annotation: dict, pdf_pages: int) -> list[dict]:
    """Page-level questions with the layout box as the evidence ``bbox``.

    ``pdf_pages`` is checked against the annotation's range: a mismatch means every page id
    of that document is off by a constant, which would look like a retrieval regression.
    """
    expected = pages_per_doc(annotation)
    if pdf_pages != expected:
        raise ValueError(
            f"page count mismatch for {annotation['doc_name']}: "
            f"pdf has {pdf_pages}, annotations say {expected}"
        )
    stem = Path(annotation["doc_name"]).stem
    questions: list[dict] = []
    for index, item in enumerate(annotation["questions"]):
        boxed = {int(mapping["page"]) for mapping in item["layout_mapping"]}
        evidence = [
            {
                "doc_id": stem,
                "page": int(mapping["page"]),
                "bbox": normalize_bbox(mapping["bbox"], mapping["page_size"]),
            }
            for mapping in item["layout_mapping"]
        ]
        # Two questions in the full set have a gold page with no box. One page-level
        # citation keeps recall honest; `iou_at_threshold` counts it as a hit by design.
        evidence += [
            {"doc_id": stem, "page": page}
            for page in sorted({int(page) for page in item["page_id"]} - boxed)
        ]
        questions.append(
            {
                "qid": f"{stem}#{index}",
                "text": item["Q"],
                "answer": str(item["A"]),
                "evidence": evidence,
                "hops": len({int(page) for page in item["page_id"]}),
            }
        )
    return questions


def _download(name: str) -> Path:
    """One Hub file. Set ``HF_ENDPOINT`` for a mirror; the error names the network, not the data."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(DATASET, name, repo_type="dataset"))


def read_annotations() -> list[dict]:
    """The annotation rows. 0.8 MB, so it is always fetched (and then cached by the Hub)."""
    return [
        json.loads(line) for line in _download(ANNOTATIONS).read_text().splitlines() if line
    ]


def extract(rar: Path, out: Path, stems: list[str]) -> None:
    """Unpack only the chosen PDFs. ``bsdtar`` reads RAR; no extra Python dependency."""
    members = [f"{ARCHIVE_PREFIX}/{stem}.pdf" for stem in stems]
    subprocess.run(["bsdtar", "-xf", str(rar), "-C", str(out), *members], check=True)


def write_benchmark(
    annotations: list[dict], pdf_dir: Path, out: Path, per_domain: int
) -> tuple[Path, Path]:
    """Copy the chosen PDFs to ``<out>/corpus`` and map their gold to ``questions.json``."""
    import pymupdf

    chosen = select_docs(annotations, per_domain)
    corpus = out / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    questions: list[dict] = []
    for annotation in chosen:
        stem = Path(annotation["doc_name"]).stem
        source = pdf_dir / f"{stem}.pdf"
        if not source.is_file():
            raise FileNotFoundError(f"missing PDF for {stem}: {source}")
        destination = corpus / source.name
        if not destination.is_file():
            destination.write_bytes(source.read_bytes())
        with pymupdf.open(source) as document:
            questions.extend(to_questions(annotation, pdf_pages=document.page_count))
    questions_path = out / "questions.json"
    questions_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    return questions_path, corpus


def main() -> None:
    """Read the annotations, unpack the chosen PDFs, write the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "artifacts-mmdocir"))
    parser.add_argument("--per-domain", type=int, default=4)
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help="directory already holding the PDFs; skips downloading and unpacking the archive",
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    annotations = read_annotations()
    stems = [
        Path(item["doc_name"]).stem for item in select_docs(annotations, args.per_domain)
    ]

    if args.pdf_dir:
        questions_path, corpus = write_benchmark(
            annotations, Path(args.pdf_dir), out, args.per_domain
        )
    else:
        with tempfile.TemporaryDirectory() as work:
            extract(_download(PDF_ARCHIVE), Path(work), stems)
            questions_path, corpus = write_benchmark(
                annotations, Path(work) / ARCHIVE_PREFIX, out, args.per_domain
            )

    questions = json.loads(questions_path.read_text())
    pages = sum(pages_per_doc(item) for item in select_docs(annotations, args.per_domain))
    print(f"corpus    {corpus}  ({len(stems)} documents, {pages} pages)")
    print(f"questions {questions_path}  ({len(questions)} questions)")


if __name__ == "__main__":
    main()
