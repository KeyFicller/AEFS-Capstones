"""Proration: share a monthly fee across the days of one month."""

from fractions import Fraction


def prorated(monthly_fee_fen: int, active_days: int, days_in_month: int) -> Fraction:
    """The exact share of `monthly_fee_fen` earned by `active_days` of the month."""
    if days_in_month <= 0:
        raise ValueError("days_in_month must be positive")
    if not 0 <= active_days <= days_in_month:
        raise ValueError("active_days must fall inside the month")
    return Fraction(monthly_fee_fen * active_days, days_in_month)
