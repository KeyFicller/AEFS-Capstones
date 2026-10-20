"""Billing dates in February, 2026."""

from datetime import date

from adapters.repo import Plan, Repo, Subscription
from api.handlers import get_invoice

ACME = Repo(
    [
        Subscription(
            tenant="acme",
            plan=Plan(code="pro", name="Pro", monthly_fee_fen=9999),
            start_date=date(2026, 2, 15),
        )
    ]
)


def test_half_month_is_prorated_and_rounded_to_the_even_fen():
    invoice = get_invoice(ACME, "acme", "2026-02")

    assert invoice["lines"][0]["amount_fen"] == 5000
    assert invoice["subtotal_fen"] == 5000
    assert invoice["total_fen"] == 5000
