"""CLI 入口：`hello-agent run "<task>"`。"""

import argparse

from hello_agent.loop import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hello-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_parser = sub.add_parser("run", help="跑一个任务")
    run_parser.add_argument("task")
    run_parser.add_argument(
        "--loop-forever", action="store_true", help="演示 turn 上限熔断"
    )

    args = parser.parse_args(argv)
    result = run(args.task, loop_forever=args.loop_forever)

    for observation in result.observations:
        print(f"[obs]  {observation}")
    print(
        f"[done] turns={result.turns} tokens={result.tokens} "
        f"cost=${result.cost:.3f} done={result.done}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
