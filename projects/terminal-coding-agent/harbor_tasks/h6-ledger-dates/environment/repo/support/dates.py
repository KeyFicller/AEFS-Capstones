"""Calendar helpers. Pure functions over `date` objects, no wall clock."""

from calendar import monthrange
from datetime import date, timedelta

ISO_DAY_LENGTH = 10


def parse_date(text: str) -> date:
    """Parse an ISO `YYYY-MM-DD` day."""
    if len(text) != ISO_DAY_LENGTH or text[4] != "-" or text[7] != "-":
        raise ValueError(f"not an ISO day: {text!r}")
    return date.fromisoformat(text)


def days_in_month(year: int, month: int) -> int:
    """Number of days in that calendar month (leap years included)."""
    return monthrange(year, month)[1]


def parse_period(text: str) -> tuple[date, date]:
    """`YYYY-MM` -> (first day of the month, first day of the next month)."""
    year, sep, month = text.partition("-")
    if not sep or len(year) != 4 or len(month) != 2:
        raise ValueError(f"not an ISO month: {text!r}")
    first = date(int(year), int(month), 1)
    if first.month == 12:
        return first, date(first.year + 1, 1, 1)
    return first, date(first.year, first.month + 1, 1)


def clamp(
    start: date,
    end: date | None,
    lower: date,
    upper: date,
) -> tuple[date, date] | None:
    """Intersect an inclusive [start, end] with a half-open [lower, upper).

    Returns None when the two ranges do not overlap.
    """
    lo = max(start, lower)
    hi = min(end + timedelta(days=1), upper) if end is not None else upper
    if lo >= hi:
        return None
    return lo, hi


def day_count(window: tuple[date, date]) -> int:
    """How many days the half-open window covers."""
    lower, upper = window
    return (upper - lower).days
