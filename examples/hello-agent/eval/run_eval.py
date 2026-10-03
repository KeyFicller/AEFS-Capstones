"""Minimal eval: run a few cases, write eval/results.jsonl, print the score.

Real projects swap CASES for a dataset, `hello` for the real harness, and add a baseline.
"""

import json
from pathlib import Path

from hello_agent.core import hello

CASES = [("a", "hello, a!"), ("b", "hello, b!"), ("c", "hello, c!")]


def main() -> None:
    rows = [{"input": name, "pass": hello(name) == expected} for name, expected in CASES]
    out = Path(__file__).with_name("results.jsonl")
    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    score = sum(row["pass"] for row in rows) / len(rows)
    print(f"pass@1={score:.2f}  -> {out}")


if __name__ == "__main__":
    main()
