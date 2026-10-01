"""Metric keys and the labels the report prints."""

COLUMN_ORDER = ("lines", "words", "chars")

METRIC_LABELS = {"lines": "Lines", "words": "Words", "chars": "Chars"}


def label_for(metric: str) -> str:
    """Human label for a metric key."""
    return METRIC_LABELS[metric]
