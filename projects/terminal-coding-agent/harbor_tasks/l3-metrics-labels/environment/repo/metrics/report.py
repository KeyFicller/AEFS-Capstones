"""Rendering of one statistics snapshot."""

from metrics.schema import COLUMN_ORDER, label_for


def render_pairs(stats: dict[str, int]) -> str:
    """One ``key=value`` pair per line, in canonical column order."""
    return "\n".join(f"{key}={stats[key]}" for key in COLUMN_ORDER)


def render_labelled(stats: dict[str, int]) -> str:
    """One ``Label: value`` pair per line, in canonical column order."""
    return "\n".join(f"{label_for(key)}: {stats[key]}" for key in COLUMN_ORDER)
