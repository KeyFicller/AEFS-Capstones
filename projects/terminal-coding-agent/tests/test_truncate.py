from terminal_coding_agent.config import CHARS_PER_TOKEN_APPROX
from terminal_coding_agent.tools.truncate import truncate


def test_under_limit_unchanged() -> None:
    assert truncate("hi", limit=10) == "hi"


def test_over_limit_adds_marker() -> None:
    limit = 2
    text = "x" * (limit * CHARS_PER_TOKEN_APPROX + 50)
    out = truncate(text, limit=limit)
    assert out.endswith(f"...[truncated, ~{limit} tokens]")
    assert len(out) < len(text)
