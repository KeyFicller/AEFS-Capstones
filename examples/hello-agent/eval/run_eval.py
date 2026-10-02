"""最小评测：跑几个任务，写 eval/results.jsonl，打印 pass@1。

真实项目把 TASKS 换成数据集、把 run 换成真实 harness、把基线换掉即可。
"""

import json
from pathlib import Path

from hello_agent.loop import run

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
