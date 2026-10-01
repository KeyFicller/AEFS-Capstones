"""Command line entry point."""

from pathlib import Path

from metrics.counter import count_all
from metrics.report import render_labelled, render_pairs


def main(argv: list[str]) -> int:
    """Print the statistics of one file, plain or labelled."""
    labelled = "--labelled" in argv
    paths = [argument for argument in argv if argument != "--labelled"]
    if len(paths) != 1:
        print("usage: python3 -m metrics [--labelled] <path>")
        return 2

    text = Path(paths[0]).read_text(encoding="utf-8")
    stats = count_all(text)
    print(render_labelled(stats) if labelled else render_pairs(stats))
    return 0
