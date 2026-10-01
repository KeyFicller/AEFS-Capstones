"""Aggregate one Harbor job directory into ``eval/results.jsonl``.

Harbor's ``result.json`` leaves ``pass_at_k`` empty and ``cost_usd`` null, so the
numbers are read straight from the trial directories instead: the verifier's
``reward.txt``, the agent's ``.agent/trace.json`` (turns / tokens / stop_reason)
and the artifact manifest.

Usage: python3 scripts/collect_eval_results.py <job-dir> [--out eval/results.jsonl]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TRIAL_SUFFIX = re.compile(r"__[A-Za-z0-9]+$")

# Reporting order and the level each task belongs to. `greeter-fix` is the
# smoke task and is deliberately excluded from the capability score.
CAPABILITY_LEVELS = {
    "l1-shipping-average": "L1",
    "l2-unit-resolution": "L2",
    "l3-metrics-labels": "L3",
    "l4-cancelled-orders": "L4",
    "l5-settings-cache": "L5",
}
SMOKE_TASKS = {"greeter-fix"}


def task_id(trial_dir: Path) -> str:
    return TRIAL_SUFFIX.sub("", trial_dir.name)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _reward(trial_dir: Path) -> int | None:
    path = trial_dir / "verifier" / "reward.txt"
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _artifact_status(trial_dir: Path) -> str:
    manifest = _read_json(trial_dir / "artifacts" / "manifest.json")
    if not manifest:
        return "missing"
    if all(entry.get("status") == "ok" for entry in manifest):
        return "ok"
    return "partial"


def collect(job_dir: Path) -> list[dict]:
    rows = []
    for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        tid = task_id(trial_dir)
        trace = _read_json(trial_dir / "artifacts" / "app" / ".agent" / "trace.json")
        rows.append(
            {
                "job": job_dir.name,
                "trial": trial_dir.name,
                "task_id": tid,
                "level": CAPABILITY_LEVELS.get(tid, "smoke" if tid in SMOKE_TASKS else "?"),
                "reward": _reward(trial_dir),
                "artifacts_status": _artifact_status(trial_dir),
                "stop_reason": trace.get("stop_reason"),
                "turns": trace.get("turns"),
                "input_tokens": trace.get("input_tokens"),
                "output_tokens": trace.get("output_tokens"),
                "cache_read_tokens": trace.get("cache_read_tokens"),
                "cost_rmb": trace.get("cost_rmb"),
            }
        )
    rows.sort(key=lambda row: (row["level"] == "smoke", row["level"], row["task_id"]))
    return rows


def report(rows: list[dict]) -> str:
    lines = [
        f"{'task':<22} {'level':<5} {'reward':<6} {'turns':<6} {'in_tok':<8} "
        f"{'out_tok':<8} {'rmb':<9} {'stop_reason':<12} artifacts",
        "-" * 100,
    ]
    for row in rows:
        lines.append(
            f"{row['task_id']:<22} {row['level']:<5} {str(row['reward']):<6} "
            f"{str(row['turns']):<6} {str(row['input_tokens']):<8} "
            f"{str(row['output_tokens']):<8} "
            f"{(row['cost_rmb'] or 0):<9.6f} "
            f"{str(row['stop_reason']):<12} {row['artifacts_status']}"
        )
    capability = [r for r in rows if r["level"] != "smoke"]
    solved = [r for r in capability if r["reward"] == 1]
    per_level = " ".join(
        f"{r['level']}={'pass' if r['reward'] == 1 else 'fail'}" for r in capability
    )
    lines += [
        "-" * 100,
        f"capability: {len(solved)}/{len(capability)} passed ({per_level})",
        f"total cost: {sum(r['cost_rmb'] or 0 for r in rows):.6f} RMB",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("eval/results.jsonl"))
    args = parser.parse_args()

    rows = collect(args.job_dir)
    if not rows:
        print(f"no trials found in {args.job_dir}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(report(rows))
    print(f"\nappended {len(rows)} row(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
