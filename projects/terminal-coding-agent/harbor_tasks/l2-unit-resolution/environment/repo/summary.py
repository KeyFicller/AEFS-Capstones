"""Report header."""

from settings import unit_of


def header(settings: dict) -> str:
    """Title line naming the unit the report is in."""
    return f"Weight report ({unit_of(settings)})"
