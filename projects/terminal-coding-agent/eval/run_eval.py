"""Minimal eval: run a few tasks, write eval/results.jsonl, print pass@1.

For a real project, swap TASKS for a dataset, run for the real harness, and replace the baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

from terminal_coding_agent.loop import run

TASKS = ["hello a", "hello b", "hello c"]


def main() -> None:
    rows = [
        {"task": task, "done": (result := run(task)).done, "turns": result.turns}
        for task in TASKS
    ]
    out = Path(__file__).with_name("results.jsonl")
    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    pass_at_1 = sum(row["done"] for row in rows) / len(rows)
    print(f"pass@1={pass_at_1:.2f}  baseline=hello-world  -> {out}")


if __name__ == "__main__":
    main()
