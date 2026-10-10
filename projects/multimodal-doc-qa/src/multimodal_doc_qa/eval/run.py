"""Run every holdout question and append one JSONL run: header, rows, summary.

Rows are flushed as they are written. A mid-run failure must not drop questions already paid for.
"""

import json
import math
import platform
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from multimodal_doc_qa.config import PROJECT_ROOT
from multimodal_doc_qa.eval.metrics import (
    bbox_hit_rate,
    iou_at_threshold,
    ndcg_at_k,
    pool_recall,
    recall_at_k,
)
from multimodal_doc_qa.schemas import Answer, Question, page_id

RESULTS_PATH = PROJECT_ROOT / "eval" / "results.jsonl"


@dataclass
class QuestionRun:
    """One question's outcome.

    ``ranked`` is the retriever's own order and is what nDCG may use. ``pool`` is the loop's
    accumulated pages and is not score-sorted.
    """

    answer: Answer | None
    ranked: list[str]
    pool: list[str]
    rounds: int
    calls: int
    tokens: int
    stop_reason: str


def _git_commit() -> str | None:
    """Short HEAD, or ``None`` outside a work tree. Never raises."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def provenance(
    settings: object,
    mode: str,
    k: int,
    n_questions: int,
    iou_threshold: float,
    retrieval_only: bool = False,
    extractor_model: str | None = None,
) -> dict:
    """Commit, models, and the IoU threshold this run used. An IoU number without it is not comparable.

    A retrieval-only run builds no chat model and measures no evidence overlap, so those
    fields are ``None``: a leftover default would be read as the value that was used. The
    exception is ``extractor_model``: the keyword arm retrieves with an LLM even in a
    retrieval-only run, so the model it used is recorded there.
    """
    return {
        "commit": _git_commit(),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": mode,
        "top_k": k,
        "iou_threshold": None if retrieval_only else iou_threshold,
        "n_questions": n_questions,
        "embedder_model": getattr(settings, "embedder_model", None),
        "ocr_embedder_model": getattr(settings, "ocr_embedder_model", None),
        "answerer_model": None if retrieval_only else getattr(settings, "answerer_model", None),
        "extractor_model": extractor_model,
        "max_rounds": None if retrieval_only else getattr(settings, "max_rounds", None),
        "max_ask_calls": None if retrieval_only else getattr(settings, "max_ask_calls", None),
        "python": platform.python_version(),
    }


def _score(
    question: Question,
    run: QuestionRun,
    k: int,
    iou_threshold: float,
    *,
    retrieval_only: bool = False,
) -> dict:
    """One per-question row: the metrics plus the raw material behind them.

    A retrieval-only run has no answer and no loop, so every graph-side field is ``None``:
    a ``0`` would read as a loop that failed rather than one that never ran.
    """
    relevant = {page_id(c.doc_id, c.page) for c in question.evidence}
    citations = list(run.answer.citations) if run.answer else []
    return {
        "kind": "question",
        "qid": question.qid,
        "hops": question.hops,
        "gold_answer": question.answer,
        "answer": None if retrieval_only else (run.answer.text if run.answer else None),
        "citations": None if retrieval_only else [c.model_dump() for c in citations],
        "ranked": run.ranked,
        "pool": None if retrieval_only else run.pool,
        "ndcg_at_k": round(ndcg_at_k(run.ranked, relevant, k), 4),
        "recall_at_k": round(recall_at_k(run.ranked, relevant, k), 4),
        "pool_recall": None if retrieval_only else round(pool_recall(run.pool, relevant), 4),
        "iou_at_threshold": (
            None
            if retrieval_only
            else round(iou_at_threshold(citations, question.evidence, threshold=iou_threshold), 4)
        ),
        "bbox_hit_rate": (
            None if retrieval_only else round(bbox_hit_rate(citations, question.evidence), 4)
        ),
        "rounds": None if retrieval_only else run.rounds,
        "calls": None if retrieval_only else run.calls,
        "tokens": None if retrieval_only else run.tokens,
        "stop_reason": None if retrieval_only else run.stop_reason,
    }


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. No interpolation on a handful of samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    # Nearest rank is ceil(fraction * N), not round(): banker's rounding picks a
    # different element and makes p50/p95 depend on the sample count's parity.
    rank = math.ceil(fraction * len(ordered))
    index = min(len(ordered) - 1, max(0, rank - 1))
    return round(ordered[index], 3)


def run_eval(
    questions: list[Question],
    run_question: Callable[[Question], QuestionRun],
    *,
    out_path: Path = RESULTS_PATH,
    settings: object | None = None,
    mode: str = "maxsim",
    k: int = 5,
    iou_threshold: float = 0.5,
    max_tokens: int | None = None,
    max_seconds: float | None = None,
    retrieval_only: bool = False,
    extractor_model: str | None = None,
) -> dict:
    """Answer every question, append rows, return the summary.

    ``retrieval_only`` runs have no answer and no loop, so the graph-side fields and the
    token cap do not apply. ``max_tokens`` and ``max_seconds`` stop the suite and keep the
    rows already written. ``stopped_after`` names the first question that was not run.
    ``extractor_model`` is the model the keyword arm retrieved with, or ``None``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latencies: list[float] = []
    tokens = 0
    stopped_after: str | None = None
    started = time.perf_counter()
    ndcgs: list[float] = []
    recalls: list[float] = []
    pool_recalls: list[float] = []
    ious: list[float] = []
    bboxes: list[float] = []
    stop_reasons: list[str] = []

    with out_path.open("a") as handle:
        header = {
            "kind": "run",
            "retrieval_only": retrieval_only,
            **(
                provenance(
                    settings,
                    mode,
                    k,
                    len(questions),
                    iou_threshold,
                    retrieval_only,
                    extractor_model,
                )
                if settings
                else {}
            ),
        }
        handle.write(json.dumps(header) + "\n")
        handle.flush()

        for question in questions:
            if not retrieval_only and max_tokens is not None and tokens >= max_tokens:
                stopped_after = question.qid
                break
            if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                stopped_after = question.qid
                break

            begin = time.perf_counter()
            run = run_question(question)
            latency = time.perf_counter() - begin

            row = _score(question, run, k, iou_threshold, retrieval_only=retrieval_only)
            row["latency_s"] = round(latency, 3)
            handle.write(json.dumps(row) + "\n")
            handle.flush()

            latencies.append(latency)
            ndcgs.append(row["ndcg_at_k"])
            recalls.append(row["recall_at_k"])
            if not retrieval_only:
                pool_recalls.append(row["pool_recall"])
                ious.append(row["iou_at_threshold"])
                bboxes.append(row["bbox_hit_rate"])
                stop_reasons.append(run.stop_reason)
                tokens += run.tokens

        summary = {
            "kind": "summary",
            "n_done": len(ndcgs),
            "n_questions": len(questions),
            "stopped_after": stopped_after,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "ndcg_at_k": _mean(ndcgs),
            "recall_at_k": _mean(recalls),
            "pool_recall": None if retrieval_only else _mean(pool_recalls),
            "iou_at_threshold": None if retrieval_only else _mean(ious),
            "bbox_hit_rate": None if retrieval_only else _mean(bboxes),
            "latency_p50_s": _percentile(latencies, 0.50),
            "latency_p95_s": _percentile(latencies, 0.95),
            "tokens": None if retrieval_only else tokens,
            "stop_reasons": None if retrieval_only else _tally(stop_reasons),
        }
        handle.write(json.dumps(summary) + "\n")

    return summary


def _tally(values: list[str]) -> dict[str, int]:
    return dict(Counter(values))
