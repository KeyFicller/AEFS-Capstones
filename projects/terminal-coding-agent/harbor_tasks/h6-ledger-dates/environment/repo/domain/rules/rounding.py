"""Invoice rounding: per line, or once on the invoice total."""

from fractions import Fraction

from config.settings import PER_INVOICE
from support.money import round_half_even


def display_lines(amounts: list[Fraction]) -> list[int]:
    """Per-line rounded values, used for presentation only."""
    return [round_half_even(amount) for amount in amounts]


def total(amounts: list[Fraction], mode: str) -> int:
    """Invoice total under the requested rounding mode."""
    if mode == PER_INVOICE:
        return round_half_even(sum(amounts, Fraction(0)))
    return sum(display_lines(amounts))
