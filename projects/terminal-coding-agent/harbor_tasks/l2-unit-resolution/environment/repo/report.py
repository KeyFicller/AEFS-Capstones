"""Totals table rendering."""

from settings import unit_of


def format_total(total: float, settings: dict) -> str:
    """One total line: the number, then the configured unit."""
    return f"{total:.2f} {unit_of(settings)}"
