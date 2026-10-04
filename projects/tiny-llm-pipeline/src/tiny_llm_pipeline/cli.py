"""Console-script entry point: `tiny-llm-pipeline <name>`."""

import argparse

from tiny_llm_pipeline.core import hello


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tiny-llm-pipeline")
    parser.add_argument("name")
    args = parser.parse_args(argv)
    print(hello(args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
