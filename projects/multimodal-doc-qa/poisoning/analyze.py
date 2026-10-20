"""Re-derive the cell table from stored answers. No model, no corpus, no tokens.

The verdict rule is wording-based and can be wrong; keeping the answers means a fixed rule
costs a rerun of this file instead of the 120 asks. Cells in the results file are derived
data, so this replaces them rather than appending a second, disagreeing set.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_asks(rows: list[dict]) -> list[dict]:
    """Keep one repeat per (condition, variant, arm, repeat), from the newest run.

    Re-running a variant appends a fresh run; keeping both would compute a rate over
    two different indexes.
    """
    if not any(row["kind"] == "run" for row in rows):
        sys.exit("no run header in the results file")

    newest: dict[tuple, dict] = {}
    for row in rows:
        if row["kind"] != "ask":
            continue
        key = (row["condition"], row["variant"], row["arm"], row["repeat"])
        newest.pop(key, None)
        newest[key] = row
    return list(newest.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    rows = load(args.results)
    asks = latest_asks(rows)
    case = harness.load_case(args.results.parent)
    # Re-classify: the rule may have changed since the ask was written. A verdict stored at
    # ask time is not evidence, the answer text is.
    for row in asks:
        row["main_class"] = harness.classify(
            row.get("answer"), case["poisoned"], case["gold"]
        )

    cells = harness.summarize(asks)
    kept = [row for row in rows if row["kind"] != "cell"]
    args.results.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [*kept, *cells])
    )
    harness.print_summary(cells, sorted({cell["arm"] for cell in cells}))
    print(f"\nre-derived {len(cells)} cells from {len(asks)} asks -> {args.results}")


if __name__ == "__main__":
    main()
