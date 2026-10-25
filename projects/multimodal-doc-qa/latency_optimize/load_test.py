"""Open-loop load test: fire requests on a fixed grid plus jitter, measure what came back.

The bench in ``bench_retrieval.py`` measures one hop with no other traffic. This one asks
whether the server holds up when turns overlap, so arrivals are scheduled *ahead of time*
and never wait for a previous request to finish. A closed loop (N workers looping on
``invoke``) would hide the queue: each worker waits for its own reply, so offered load
silently drops exactly when the server is slow.

Every row separates the two clocks the server cannot tell apart:

- ``wall`` -- what the client waited, the number a user feels.
- ``server`` -- ``latency_s`` from the ``/ask`` body, measured inside ``answer_question``.
- ``queue`` -- ``wall - server``, i.e. time spent before the handler ran, or after it
  returned. That is the threadpool / accept queue, and it is the first thing to grow when
  a process is saturated.

Arrivals are ``i * interval + U(-jitter, +jitter) * interval`` on the absolute grid, so a
jittered arrival cannot push the mean rate off target.

    python latency_optimize/load_test.py --rate 1 --jitter 0.3 --count 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


@dataclass
class Row:
    """One request: when it was due, when it left, and what the two sides measured."""

    index: int
    scheduled: float
    dispatched: float = 0.0
    finished: float = 0.0
    status: int = 0
    server_s: float = 0.0
    stop_reason: str = ""
    rounds: int = 0
    calls: int = 0
    tokens: int = 0
    error: str = ""
    admission_wait: float = 0.0


@dataclass
class Plan:
    """What to send, and how fast."""

    base_url: str
    questions: list[str]
    rate: float
    jitter: float
    count: int
    timeout: float
    max_inflight: int
    mode: str | None = None
    seed: int = 7
    # Set by ``drive`` so a row can store offsets instead of absolute perf_counter values.
    window_start: float = 0.0


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)

    def ok(self) -> list[Row]:
        return [row for row in self.rows if row.status == 200]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(len(ordered) * fraction + 0.999)))
    return round(ordered[rank - 1], 3)


async def one(client: httpx.AsyncClient, plan: Plan, row: Row, gate: asyncio.Semaphore) -> None:
    """Send one turn and record both clocks. Never raises; a failure is a row."""
    waited_from = time.perf_counter()
    await gate.acquire()
    row.admission_wait = round(time.perf_counter() - waited_from, 4)
    row.dispatched = time.perf_counter() - plan.window_start
    body = {"question": plan.questions[row.index % len(plan.questions)]}
    if plan.mode:
        body["mode"] = plan.mode
    try:
        response = await client.post(f"{plan.base_url}/ask", json=body)
        row.status = response.status_code
        if response.status_code == 200:
            payload = response.json()
            row.server_s = payload["latency_s"]
            row.stop_reason = payload["stop_reason"]
            row.rounds = payload["rounds"]
            row.calls = payload["calls"]
            row.tokens = payload["tokens"]
        else:
            row.error = response.text[:200]
    except Exception as exc:  # noqa: BLE001 - a transport failure is a result, not a crash
        row.error = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        row.finished = time.perf_counter() - plan.window_start
        gate.release()


async def drive(plan: Plan) -> Report:
    """Fire ``plan.count`` arrivals on the jittered grid, then wait for what is in flight."""
    interval = 1.0 / plan.rate
    rng = random.Random(plan.seed)
    plan.window_start = time.perf_counter()
    rows = [Row(index=i, scheduled=i * interval + rng.uniform(-plan.jitter, plan.jitter) * interval) for i in range(plan.count)]
    gate = asyncio.Semaphore(plan.max_inflight)

    limits = httpx.Limits(max_connections=plan.max_inflight, max_keepalive_connections=plan.max_inflight)
    async with httpx.AsyncClient(timeout=plan.timeout, limits=limits) as client:
        tasks = []
        for row in rows:
            delay = plan.window_start + row.scheduled - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(client, plan, row, gate)))
        await asyncio.gather(*tasks)
    return Report(rows=rows)


def summarize(plan: Plan, report: Report) -> dict:
    """One dict for the table, one for the JSON. Percentiles on the rows that answered."""
    done = report.ok()
    walls = [row.finished - row.dispatched for row in done]
    servers = [row.server_s for row in done]
    queues = [max(0.0, (row.finished - row.dispatched) - row.server_s) for row in done]
    lag = [row.dispatched - row.scheduled for row in report.rows]
    span = max((row.finished for row in report.rows), default=0.0)
    return {
        "rate_target": plan.rate,
        "jitter": plan.jitter,
        "count": plan.count,
        "ok": len(done),
        "failed": len(report.rows) - len(done),
        "span_s": round(span, 2),
        "achieved_rps": round(len(done) / span, 3) if span else 0.0,
        "offered_rps": round(plan.rate, 3),
        "wall_p50": percentile(walls, 0.50),
        "wall_p95": percentile(walls, 0.95),
        "wall_p99": percentile(walls, 0.99),
        "wall_max": round(max(walls), 3) if walls else 0.0,
        "server_p50": percentile(servers, 0.50),
        "server_p95": percentile(servers, 0.95),
        "queue_p50": percentile(queues, 0.50),
        "queue_p95": percentile(queues, 0.95),
        "arrival_lag_max": round(max(lag), 3) if lag else 0.0,
        "admission_wait_max": round(max((row.admission_wait for row in report.rows), default=0.0), 3),
        "max_inflight_hit": any(row.admission_wait > 0.001 for row in report.rows),
        "tokens": sum(row.tokens for row in done),
        "calls": sum(row.calls for row in done),
        "stop_reasons": _tally([row.stop_reason for row in done]),
        "errors": _tally([row.error for row in report.rows if row.error]),
    }


def _tally(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def load_questions(limit: int) -> list[str]:
    """Question texts from the artifact catalog, one per request in rotation."""
    path = ARTIFACTS / "questions.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    texts = [row["text"] for row in rows if row.get("text")]
    if not texts:
        raise SystemExit(f"no question text in {path}")
    return texts[:limit] if limit else texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rate", type=float, default=1.0, help="arrivals per second")
    parser.add_argument("--jitter", type=float, default=0.3, help="fraction of the interval, +/-")
    parser.add_argument("--count", type=int, default=24, help="total requests")
    parser.add_argument("--questions", type=int, default=8, help="distinct questions, cycled")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-inflight", type=int, default=256)
    parser.add_argument("--mode", default=None, help="retrieval arm; server default when omitted")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    plan = Plan(
        base_url=args.base_url,
        questions=load_questions(args.questions),
        rate=args.rate,
        jitter=args.jitter,
        count=args.count,
        timeout=args.timeout,
        max_inflight=args.max_inflight,
        mode=args.mode,
    )
    report = asyncio.run(drive(plan))
    summary = summarize(plan, report)

    print(f"\nrate {summary['rate_target']}/s +/-{summary['jitter']:.0%}  "
          f"{summary['count']} requests, {summary['ok']} ok, {summary['failed']} failed")
    print(f"span {summary['span_s']}s  achieved {summary['achieved_rps']}/s  "
          f"max queue {summary['queue_p95']}s  max arrival lag {summary['arrival_lag_max']}s")
    print(f"{'':16s}{'p50':>8s}{'p95':>8s}{'p99':>8s}{'max':>8s}")
    print(f"{'wall (s)':16s}{summary['wall_p50']:8.2f}{summary['wall_p95']:8.2f}"
          f"{summary['wall_p99']:8.2f}{summary['wall_max']:8.2f}")
    print(f"{'server (s)':16s}{summary['server_p50']:8.2f}{summary['server_p95']:8.2f}")
    print(f"{'queue (s)':16s}{summary['queue_p50']:8.2f}{summary['queue_p95']:8.2f}")
    if summary["stop_reasons"]:
        print(f"stop reasons   {summary['stop_reasons']}")
    if summary["errors"]:
        print(f"errors         {summary['errors']}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "rows": [
                {key: value for key, value in row.__dict__.items()} for row in report.rows
            ],
        }
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
