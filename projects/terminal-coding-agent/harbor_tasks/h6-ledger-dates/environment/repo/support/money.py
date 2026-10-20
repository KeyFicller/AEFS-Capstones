"""Money helpers: integer fen in, integer fen out."""

from fractions import Fraction


def round_half_even(value: Fraction | int) -> int:
    """Round to the nearest fen; exact .5 goes to the even fen (Python's `round`)."""
    return int(round(Fraction(value)))


def format_fen(fen: int) -> str:
    """Render fen as a `1.23` string without touching floats."""
    sign = "-" if fen < 0 else ""
    magnitude = abs(int(fen))
    return f"{sign}{magnitude // 100}.{magnitude % 100:02d}"
