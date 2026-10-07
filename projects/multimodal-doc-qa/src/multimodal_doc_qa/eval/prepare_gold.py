"""One-off: turn a MMLongBench-Doc subset into questions.json plus a corpus directory.

Run: ``python -m multimodal_doc_qa.eval.prepare_gold --docs a.pdf,b.pdf``.
"""

import argparse
import ast
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

from multimodal_doc_qa.config import PROJECT_ROOT

DATASET = "yubo2333/MMLongBench-Doc"
PARQUET = "data/train-00000-of-00001.parquet"
NOT_ANSWERABLE = "Not answerable"


def _evidence_pages(raw: object) -> list[int]:
    """The parquet stores the page list as a string: ``"[19, 20]"``; empty is ``"[]"``."""
    if isinstance(raw, str):
        raw = ast.literal_eval(raw) if raw.strip() else []
    return [int(page) for page in (raw or [])]


def rows_to_questions(rows: list[dict], doc_ids: set[str], limit: int) -> list[dict]:
    """Page-level questions for the chosen documents.

    ``limit`` caps the questions taken from each document, so every chosen document is
    represented: the rows are grouped by document, so a global cap would drop whole ones.
    ``evidence_pages`` is a stringified list in the parquet (``"[19, 20]"``, empty ``"[]"``).
    It is 1-based; the project's page index is 0-based. Rows with no gold page
    (``Not answerable`` / empty evidence) are dropped: recall is undefined for them.
    """
    questions: list[dict] = []
    per_doc: dict[str, int] = {}
    for row in rows:
        doc_id = row["doc_id"]
        if doc_id not in doc_ids or per_doc.get(doc_id, 0) >= limit:
            continue
        pages = _evidence_pages(row.get("evidence_pages"))
        if row["answer"] == NOT_ANSWERABLE or not pages:
            continue
        stem = Path(doc_id).stem
        questions.append(
            {
                "qid": f"{stem}#{per_doc.get(doc_id, 0)}",
                "text": row["question"],
                "answer": str(row["answer"]),
                "evidence": [{"doc_id": stem, "page": page - 1} for page in pages],
                "hops": 1,
            }
        )
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
    return questions


def main() -> None:
    """Read the annotations, write questions.json, download the chosen PDFs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", required=True, help="comma-separated doc_ids (file names)")
    parser.add_argument("--questions", default=str(PROJECT_ROOT / "artifacts" / "questions.json"))
    parser.add_argument("--corpus", default=str(PROJECT_ROOT / "artifacts" / "corpus"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    doc_ids = {name.strip() for name in args.docs.split(",") if name.strip()}

    annotations = snapshot_download(DATASET, repo_type="dataset", allow_patterns=[PARQUET])
    rows = pq.read_table(Path(annotations) / PARQUET).to_pylist()
    questions = rows_to_questions(rows, doc_ids, limit=args.limit)

    out = Path(args.questions)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")

    corpus = Path(args.corpus)
    corpus.mkdir(parents=True, exist_ok=True)
    for name in doc_ids:
        local = hf_hub_download(DATASET, f"documents/{name}", repo_type="dataset")
        (corpus / name).write_bytes(Path(local).read_bytes())

    print(f"questions {out}  ({len(questions)} questions)")
    print(f"corpus    {corpus}  ({len(doc_ids)} documents)")


if __name__ == "__main__":
    main()
