"""Hidden checks for l3-metrics-labels. Never copied into the agent's repo."""

import subprocess
import sys
from pathlib import Path

SAMPLE = Path("/app/sample.txt")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "metrics", *args],
        cwd="/app",
        capture_output=True,
        text=True,
    )


def test_plain_mode_exits_zero():
    assert _run("sample.txt").returncode == 0


def test_plain_mode_prints_every_metric():
    text = SAMPLE.read_text(encoding="utf-8")
    assert _run("sample.txt").stdout.splitlines() == [
        f"lines={len(text.splitlines())}",
        f"words={len(text.split())}",
        f"chars={len(text)}",
    ]


def test_labelled_mode_prints_every_metric():
    text = SAMPLE.read_text(encoding="utf-8")
    assert _run("--labelled", "sample.txt").stdout.splitlines() == [
        f"Lines: {len(text.splitlines())}",
        f"Words: {len(text.split())}",
        f"Chars: {len(text)}",
    ]


def test_count_all_returns_exactly_the_three_documented_keys():
    from metrics.counter import count_all

    assert count_all("a b\nc\n") == {"lines": 2, "words": 3, "chars": 6}
