"""`current_time` output, driven by an injected clock instead of the wall clock."""

from datetime import UTC, datetime

import pytest
from terminal_coding_agent.tools import clock


@pytest.fixture
def frozen(monkeypatch):
    """Fix the clock at a UTC Tuesday; assertions stay machine-timezone independent."""
    moment = datetime(2026, 10, 6, 12, 30, 45, tzinfo=UTC)
    monkeypatch.setattr(clock, "_now", lambda: moment)
    return moment


def test_output_has_utc_local_and_epoch_lines(frozen) -> None:
    out = clock.build_current_time().invoke({})
    lines = out.splitlines()

    assert len(lines) == 3
    assert lines[0].startswith("utc:")
    assert lines[1].startswith("local:")
    assert lines[2].startswith("epoch:")


def test_the_utc_line_is_the_injected_instant(frozen) -> None:
    out = clock.build_current_time().invoke({})
    assert "2026-10-06T12:30:45+00:00" in out


def test_the_weekday_is_named(frozen) -> None:
    out = clock.build_current_time().invoke({})
    # 2026-10-06 is a Tuesday.
    assert out.count("Tuesday") == 2


def test_epoch_is_integer_seconds(frozen) -> None:
    out = clock.build_current_time().invoke({})
    assert f"epoch: {int(frozen.timestamp())}" in out


def test_two_calls_agree(frozen) -> None:
    tool = clock.build_current_time()
    assert tool.invoke({}) == tool.invoke({})


def test_the_tool_takes_no_arguments(frozen) -> None:
    assert clock.build_current_time().args == {}


def test_a_non_utc_local_offset_is_reported(monkeypatch) -> None:
    """The local line carries its own offset, whatever the machine's zone is."""
    from datetime import timedelta, timezone

    fixed = datetime(2026, 10, 6, 12, 30, 45, tzinfo=timezone(timedelta(hours=8)))
    monkeypatch.setattr(clock, "_now", lambda: fixed)

    out = clock.build_current_time().invoke({})

    assert "local: 2026-10-06T12:30:45+08:00" in out
