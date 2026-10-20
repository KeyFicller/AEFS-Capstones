"""Invoice assembly: rounding, credits and persistence of the carry-over."""

from adapters.repo import Repo
from config.settings import Settings, from_env
from domain.policy import build_lines
from domain.rules.credits import apply_credit
from domain.rules.rounding import display_lines, total
from support.dates import parse_period


def build_invoice(repo: Repo, tenant: str, period: str, settings: Settings | None = None) -> dict:
    """Build (and persist the credit carry-over of) one monthly invoice."""
    if not isinstance(period, str):
        raise TypeError("period must be a 'YYYY-MM' string")
    settings = settings or from_env()

    subscription = repo.get(tenant)
    period_start, period_end = parse_period(period)

    labelled = build_lines(subscription, period_start, period_end)
    amounts = [amount for _, amount in labelled]

    subtotal_fen = total(amounts, settings.rounding_mode)
    applied, carried = apply_credit(subtotal_fen, repo.credit_balance(tenant))
    repo.set_credit_balance(tenant, carried)

    shown = display_lines(amounts)
    return {
        "tenant": tenant,
        "period": period,
        "rounding": settings.rounding_mode,
        "lines": [
            {"label": label, "amount_fen": fen}
            for (label, _), fen in zip(labelled, shown)
        ],
        "subtotal_fen": subtotal_fen,
        "credit_applied_fen": applied,
        "credit_carried_fen": carried,
        "total_fen": subtotal_fen - applied,
    }
