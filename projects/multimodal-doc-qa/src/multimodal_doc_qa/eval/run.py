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


def provenance(settings: object, mode: str, k: int, n_questions: int, iou_threshold: float) -> dict:
    """Commit, models, and the IoU threshold this run used. An IoU number without it is not comparable."""
    return {
        "commit": _git_commit(),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": mode,
        "top_k": k,
        "iou_threshold": iou_threshold,
        "n_questions": n_questions,
        "embedder_model": getattr(settings, "embedder_model", None),
        "ocr_embedder_model": getattr(settings, "ocr_embedder_model", None),
        "answerer_model": getattr(settings, "answerer_model", None),
        "max_rounds": getattr(settings, "max_rounds", None),
        "max_ask_calls": getattr(settings, "max_ask_calls", None),
        "python": platform.python_version(),
    }


def _score(question: Question, run: QuestionRun, k: int, iou_threshold: float) -> dict:
    """One per-question row: the metrics plus the raw material behind them."""
    relevant = {page_id(c.doc_id, c.page) for c in question.evidence}
    citations = list(run.answer.citations) if run.answer else []
    return {
        "kind": "question",
        "qid": question.qid,
        "hops": question.hops,
        "gold_answer": question.answer,
        "answer": run.answer.text if run.answer else None,
        "citations": [c.model_dump() for c in citations],
        "ranked": run.ranked,
        "pool": run.pool,
        "ndcg_at_k": round(ndcg_at_k(run.ranked, relevant, k), 4),
        "recall_at_k": round(recall_at_k(run.ranked, relevant, k), 4),
        "pool_recall": round(pool_recall(run.pool, relevant), 4),
        "iou_at_threshold": round(
            iou_at_threshold(citations, question.evidence, threshold=iou_threshold), 4
        ),
        "bbox_hit_rate": round(bbox_hit_rate(citations, question.evidence), 4),
        "rounds": run.rounds,
        "calls": run.calls,
        "tokens": run.tokens,
        "stop_reason": run.stop_reason,
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
    mode: str = "vision",
    k: int = 5,
    iou_threshold: float = 0.5,
    max_tokens: int | None = None,
    max_seconds: float | None = None,
) -> dict:
    """Answer every question, append rows, return the summary.

    ``max_tokens`` and ``max_seconds`` stop the suite and keep the rows already written.
    ``stopped_after`` names the first question that was not run.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latencies: list[float] = []
    tokens = 0
    stopped_after: str | None = None
    started = time.perf_counter()
    metrics: list[tuple[float, float, float]] = []
    recalls: list[float] = []
    pool_recalls: list[float] = []
    stop_reasons: list[str] = []

    with out_path.open("a") as handle:
        header = {
            "kind": "run",
            **(provenance(settings, mode, k, len(questions), iou_threshold) if settings else {}),
        }
        handle.write(json.dumps(header) + "\n")
        handle.flush()

        for question in questions:
            if max_tokens is not None and tokens >= max_tokens:
                stopped_after = question.qid
                break
            if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                stopped_after = question.qid
                break

            begin = time.perf_counter()
            run = run_question(question)
            latency = time.perf_counter() - begin

            row = _score(question, run, k, iou_threshold)
            row["latency_s"] = round(latency, 3)
            handle.write(json.dumps(row) + "\n")
            handle.flush()

            latencies.append(latency)
            metrics.append((row["ndcg_at_k"], row["iou_at_threshold"], row["bbox_hit_rate"]))
            recalls.append(row["recall_at_k"])
            pool_recalls.append(row["pool_recall"])
            stop_reasons.append(run.stop_reason)
            tokens += run.tokens

        summary = {
            "kind": "summary",
            "n_done": len(metrics),
            "n_questions": len(questions),
            "stopped_after": stopped_after,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "ndcg_at_k": _mean([n for n, _, _ in metrics]),
            "recall_at_k": _mean(recalls),
            "pool_recall": _mean(pool_recalls),
            "iou_at_threshold": _mean([i for _, i, _ in metrics]),
            "bbox_hit_rate": _mean([b for _, _, b in metrics]),
            "latency_p50_s": _percentile(latencies, 0.50),
            "latency_p95_s": _percentile(latencies, 0.95),
            "tokens": tokens,
            "stop_reasons": _tally(stop_reasons),
        }
        handle.write(json.dumps(summary) + "\n")

    return summary


def _tally(values: list[str]) -> dict[str, int]:
    return dict(Counter(values))
