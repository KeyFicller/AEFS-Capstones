"""Append one web-check row per stored P2 ask: can DDG snippets refute the answer?

Reads answers already written by ``run.py``; no ask is re-run and no index is touched.
Rows are ``kind="webcheck"`` and keyed by the same four fields as their ask, so a second
invocation adds nothing. The verdict never feeds back into synthesis: ``RSR`` cannot move.
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze  # noqa: E402 - sibling, on sys.path because this file is the script
import harness  # noqa: E402 - parent directory, inserted above
import web  # noqa: E402 - parent directory, inserted above
from multimodal_doc_qa.config import ENV_PATH, Settings, load_local_env  # noqa: E402
from multimodal_doc_qa.synth.answer import build_chat_model  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results.jsonl"
WORK_NAME = "work"
KEY = ("condition", "variant", "arm", "repeat")


def load_ocr_index(settings) -> dict:
    """The OCR chunk index, read once. Page text is where the entity lives when the answer
    is a bare figure: a query built from the question alone cannot name the city."""
    import torch

    return torch.load(settings.ocr_index_path, weights_only=True)


def existing(rows: list[dict]) -> set[tuple]:
    """Keys already checked. A rerun must not append a second opinion per ask."""
    return {
        tuple(row[field] for field in KEY)
        for row in rows
        if row.get("kind") == "webcheck"
    }


def summarize(records: list[dict]) -> list[dict]:
    """Per (condition, variant, arm) counts. ``error`` is its own column, never ``silent``."""
    groups: dict[tuple, list[dict]] = {}
    for row in records:
        groups.setdefault((row["condition"], row["variant"], row["arm"]), []).append(row)
    out = []
    for (condition, variant, arm), items in groups.items():
        n = len(items)
        clean = [row for row in items if not row.get("web_error")]
        out.append({
            "condition": condition,
            "variant": variant,
            "arm": arm,
            "n": n,
            "error": len(items) - len(clean),
            "refutes": sum(1 for row in clean if row.get("web_refutes")),
            "refutes_body": sum(1 for row in clean if row.get("web_refutes_in") == "body"),
            "supports": sum(1 for row in clean if row.get("web_supports")),
            "silent": sum(1 for row in clean if row.get("web_silent")),
            "fetched": sum(1 for row in clean if row.get("web_fetched")),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--limit", type=int, default=None, help="check only the first N asks")
    parser.add_argument("--sleep", type=float, default=2.0, help="seconds between DDG requests")
    args = parser.parse_args()

    load_local_env(ENV_PATH)
    settings = Settings()
    case = harness.load_case(ROOT)
    rows = analyze.load(args.results)
    asks = analyze.latest_asks(rows)
    done = existing(rows)
    todo = [
        row for row in asks
        if row.get("answer") and tuple(row[field] for field in KEY) not in done
    ]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"{len(asks)} asks, {len(done)} already checked, {len(todo)} to check")
    if not todo:
        print("nothing to do")
        return

    chat = build_chat_model(settings)
    index = load_ocr_index(settings)
    records: list[dict] = []
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with harness.exclusive(ROOT / WORK_NAME / ".lock"), args.results.open("a") as handle:
        for index_row, row in enumerate(todo, 1):
            number = web.answer_number(row["answer"])
            context, context_pages = web.page_context(index, row.get("pool") or [])
            queries, tokens, error = web.extract_queries(
                chat,
                row["question"],
                row["answer"],
                context=context,
                blocked=[value for value in (number, case["gold"]) if value],
            )
            hits, search_error = web.search(queries)
            bodies, fetched_urls, fetch_errors = web.fetch_pages(hits)
            verdict = web.judge(hits, gold=case["gold"], number=number, bodies=bodies)
            record = {
                "kind": "webcheck",
                **{field: row[field] for field in KEY},
                "qid": row.get("qid"),
                "answer_number": number,
                "context_pages": context_pages,
                "context_chars": len(context),
                "web_queries": queries,
                "web_hits": len(hits),
                "web_fetched": len(fetched_urls),
                "web_fetch_urls": fetched_urls,
                "web_fetch_errors": fetch_errors[:5],
                "web_body_chars": sum(len(body) for body in bodies),
                "web_refutes": verdict["refutes"],
                "web_refutes_in": verdict["refutes_in"],
                "web_supports": verdict["supports"],
                "web_silent": verdict["silent"],
                "web_evidence": verdict["evidence"],
                "web_urls": verdict["urls"][:10],
                "web_error": error or search_error,
                "web_calls": 1,
                "web_tokens": tokens,
                "web_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
            print(
                f"  {index_row}/{len(todo)} {record['variant']:20} {record['arm']:11} "
                f"ctx={len(context_pages)}p q={len(queries)} hits={record['web_hits']} "
                f"fetch={record['web_fetched']} refutes={record['web_refutes']}"
                f"/{record['web_refutes_in']} supports={record['web_supports']} "
                f"{queries} {record['web_error'] or ''}"
            )
            if index_row < len(todo):
                time.sleep(args.sleep)

    cells = summarize(records)
    header = (
        f"{'cond':5} {'variant':20} {'arm':11} {'n':>3} {'fetched':>7} "
        f"{'refutes':>9} {'body':>6} {'supports':>9} {'silent':>8} {'error':>6}"
    )
    print(f"\n{header}\n{'-' * len(header)}")
    for cell in cells:
        n = cell["n"]
        print(
            f"{cell['condition']:5} {cell['variant']:20} {cell['arm']:11} {n:>3} "
            f"{cell['fetched']:>4}/{n:<3} {cell['refutes']:>4}/{n:<4} "
            f"{cell['refutes_body']:>3}/{n:<3} {cell['supports']:>4}/{n:<4} "
            f"{cell['silent']:>3}/{n:<4} {cell['error']:>3}/{n:<3}"
        )
    print(f"\nrows -> {args.results}")


if __name__ == "__main__":
    main()
