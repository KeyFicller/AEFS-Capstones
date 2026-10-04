"""Evidence overlay and the Streamlit viewer.

Reads ``doc-qa eval`` results, or ``turns.jsonl`` written by ``ask`` / the REPL.
Gold boxes are drawn only when ``questions.json`` is there. A citation with no
bbox is a border, not a box: no region was located.
"""

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

from multimodal_doc_qa.schemas import Answer, Citation

GOLD = (30, 110, 220)
CITED = (215, 40, 40)
PAGE_LEVEL = (230, 150, 20)

MIN_OUTLINE = 2
OUTLINE_PER_PIXEL = 300


def _outline_width(image: Image.Image) -> int:
    """Outline thick enough to see on a 2048px page and not a slab on a thumbnail."""
    return max(MIN_OUTLINE, round(max(image.size) / OUTLINE_PER_PIXEL))


def draw_citations(
    image: Image.Image,
    citations: list[Citation],
    *,
    doc_id: str,
    page: int,
    color: tuple[int, int, int] = CITED,
) -> Image.Image:
    """Copy of ``image`` with this page's citations drawn. Other pages are skipped.

    Call twice, ``GOLD`` then ``CITED``, to overlay ground truth and the model's boxes.
    """
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    outline = _outline_width(canvas)

    for citation in citations:
        if citation.doc_id != doc_id or citation.page != page:
            continue
        if citation.bbox is None:
            draw.rectangle(
                [outline, outline, width - 1 - outline, height - 1 - outline],
                outline=PAGE_LEVEL,
                width=outline,
            )
            continue
        draw.rectangle(
            [
                citation.bbox.x0 * width,
                citation.bbox.y0 * height,
                citation.bbox.x1 * width,
                citation.bbox.y1 * height,
            ],
            outline=color,
            width=outline,
        )
    return canvas


def append_turn(
    path: Path,
    *,
    mode: str,
    question: str,
    answer: Answer | None,
    rounds: int,
    calls: int,
    tokens: int,
    stop_reason: str,
) -> None:
    """One REPL / ``ask`` turn, in the same JSONL shape ``load_runs`` already reads.

    No gold, no metrics. Each turn is its own run so a repeated question does not
    overwrite the previous one inside a single header.
    """
    citations = [] if answer is None else [c.model_dump() for c in answer.citations]
    rows = (
        {"kind": "run", "mode": mode},
        {
            "kind": "question",
            "qid": question,
            "answer": None if answer is None else answer.text,
            "citations": citations,
            "rounds": rounds,
            "calls": calls,
            "tokens": tokens,
            "stop_reason": stop_reason,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_runs(path: Path) -> list[dict]:
    """One dict per run: ``header``, ``questions``, ``summary``.

    Split on ``kind="run"``. A run that never got its summary is kept with ``summary=None``.
    """
    if not path.is_file():
        return []

    runs: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        # The viewer exists to read aborted runs; a half-written line must not kill the load.
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = row.get("kind")
        if kind == "run":
            runs.append({"header": row, "questions": {}, "summary": None})
            continue
        if not runs and kind in {"question", "summary"}:
            runs.append({"header": {}, "questions": {}, "summary": None})
        if kind == "question":
            runs[-1]["questions"][row["qid"]] = row
        elif kind == "summary":
            runs[-1]["summary"] = row
    return runs


def runs_for_question(runs: list[dict], qid: str) -> list[tuple[dict, dict]]:
    """``(run, row)`` pairs where this question was answered. A truncated run is skipped."""
    return [(run, run["questions"][qid]) for run in runs if qid in run["questions"]]


def question_ids(runs: list[dict]) -> list[str]:
    """Every question any run answered, in first-seen order."""
    return list(dict.fromkeys(qid for run in runs for qid in run["questions"]))


def page_image(render_dir: Path, doc_id: str, page: int) -> Image.Image | None:
    """The rendered page, or ``None`` when it was never rendered.

    Loads a detached copy: ``Image.open`` is lazy and the handle must not depend on
    the caller to release it.
    """
    path = render_dir / doc_id / f"p{page:03d}.png"
    if not path.is_file():
        return None
    with Image.open(path) as image:
        return image.copy()


def evidence_pages(citations: list[Citation], gold: list[Citation]) -> list[tuple[str, int]]:
    """Pages either side points at, in first-seen order."""
    return list(dict.fromkeys((c.doc_id, c.page) for c in [*gold, *citations]))


def render_app() -> None:
    """Streamlit entrypoint: ``streamlit run .../viewer.py``."""
    import streamlit as st

    from multimodal_doc_qa.config import Settings, artifact_paths

    settings = Settings()

    st.set_page_config(page_title="multimodal-doc-qa viewer", layout="wide")
    st.title("multimodal-doc-qa — evidence viewer")

    with st.sidebar:
        st.header("Inputs")
        artifacts = Path(st.text_input("Artifacts directory", value=str(settings.artifacts_dir)))
        results = [
            path
            for path in (
                st.text_input(
                    "vision results.jsonl", value=os.environ.get("MDQ_RESULTS_VISION", "")
                ),
                st.text_input("ocr results.jsonl", value=os.environ.get("MDQ_RESULTS_OCR", "")),
            )
            if path
        ]
        questions_path = artifacts / "questions.json"
        show_gold = (
            st.checkbox("Show gold evidence", value=True) if questions_path.is_file() else False
        )

    if not results and (artifacts / "turns.jsonl").is_file():
        results = [str(artifacts / "turns.jsonl")]

    runs = [run for path in results for run in load_runs(Path(path))]
    if not runs:
        st.info(
            "No turns yet. Ask in `doc-qa`, or point the sidebar at a `results.jsonl` from `doc-qa eval`."
        )
        return

    render_dir = artifact_paths(artifacts)[0]
    qids = question_ids(runs)
    qid = st.selectbox("Question", qids)

    gold: list[Citation] = []
    if questions_path.is_file():
        raw = json.loads(questions_path.read_text())
        gold = [
            Citation.model_validate(c)
            for question in raw
            if question["qid"] == qid
            for c in question["evidence"]
        ]
        st.caption(next((q["text"] for q in raw if q["qid"] == qid), ""))

    paired = runs_for_question(runs, qid)
    columns = st.columns(max(1, len(paired)))
    for column, (run, row) in zip(columns, paired, strict=False):
        with column:
            mode = run["header"].get("mode", "?")
            st.subheader(mode)
            st.write(row.get("answer") or "_no answer_")
            bits = [
                f"{label} {row[key]}"
                for label, key in (
                    ("nDCG@k", "ndcg_at_k"),
                    ("IoU@τ", "iou_at_threshold"),
                    ("strict", "bbox_hit_rate"),
                )
                if row.get(key) is not None
            ]
            bits += [
                f"{label} {row.get(key) or '-'}"
                for label, key in (
                    ("rounds", "rounds"),
                    ("calls", "calls"),
                    ("stop_reason", "stop_reason"),
                )
                if key in row
            ]
            if bits:
                st.caption("  ".join(bits))

    citations = [Citation.model_validate(c) for _, row in paired for c in row.get("citations", [])]
    for doc_id, page in evidence_pages(citations, gold):
        image = page_image(render_dir, doc_id, page)
        st.markdown(f"**{doc_id} page {page}**")
        if image is None:
            st.warning(f"page not rendered: {render_dir / doc_id / f'p{page:03d}.png'}")
            continue
        if show_gold:
            image = draw_citations(image, gold, doc_id=doc_id, page=page, color=GOLD)
        image = draw_citations(image, citations, doc_id=doc_id, page=page, color=CITED)
        st.image(image, caption="blue = gold evidence, red = cited, amber = no region claimed")
        st.caption(
            " | ".join(
                "page-level" if c.bbox is None else "box"
                for c in citations
                if c.doc_id == doc_id and c.page == page
            )
            or "no citation on this page"
        )


if __name__ == "__main__":
    render_app()
