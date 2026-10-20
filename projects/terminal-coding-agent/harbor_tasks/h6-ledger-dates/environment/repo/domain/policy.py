"""Turn a subscription into the exact (unrounded) charge lines of one month."""

from datetime import date

from adapters.repo import Subscription
from domain.rules.proration import prorated
from support.dates import clamp, day_count, days_in_month


def _intersect(left, right):
    lower = max(left[0], right[0])
    upper = min(left[1], right[1])
    return None if lower >= upper else (lower, upper)


def _amount(fee_fen: int, window, period_start: date):
    if window is None:
        return None
    return prorated(fee_fen, day_count(window), days_in_month(period_start.year, period_start.month))


def build_lines(subscription: Subscription, period_start: date, period_end: date) -> list[tuple[str, int]]:
    """Charge lines in invoice order: the plan first, then addons by code."""
    subscription_window = clamp(
        subscription.start_date, subscription.end_date, period_start, period_end
    )
    lines: list[tuple[str, int]] = []

    base = _amount(subscription.plan.monthly_fee_fen, subscription_window, period_start)
    if base is not None:
        lines.append((subscription.plan.name, base))

    for addon in sorted(subscription.addons, key=lambda item: item.code):
        addon_window = clamp(addon.start_date, addon.end_date, period_start, period_end)
        if addon_window is None or subscription_window is None:
            continue
        window = _intersect(addon_window, subscription_window)
        amount = _amount(addon.monthly_fee_fen, window, period_start)
        if amount is not None:
            lines.append((addon.code, amount))

    return lines
