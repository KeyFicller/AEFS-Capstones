"""Console-script entry point: `hello-agent <name>`."""

import argparse

from hello_agent.core import hello


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hello-agent")
    parser.add_argument("name")
    args = parser.parse_args(argv)
    print(hello(args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
