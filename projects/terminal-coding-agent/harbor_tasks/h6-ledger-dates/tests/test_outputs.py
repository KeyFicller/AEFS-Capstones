"""Hidden checks for h6-ledger-dates. Never copied into the agent's repo."""

from datetime import date

import pytest
from adapters.repo import Addon, Plan, Repo, Subscription
from api.handlers import get_invoice
from config.settings import PER_INVOICE, PER_LINE, Settings, from_env


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LEDGER_ROUNDING_MODE", raising=False)


def _repo(
    monthly_fee_fen: int,
    start: date,
    end: date | None = None,
    addons=(),
    credits: dict | None = None,
) -> Repo:
    return Repo(
        [
            Subscription(
                tenant="acme",
                plan=Plan(code="pro", name="Pro", monthly_fee_fen=monthly_fee_fen),
                start_date=start,
                end_date=end,
                addons=tuple(addons),
            )
        ],
        credits=credits,
    )


def test_full_month_is_charged_in_full():
    invoice = get_invoice(_repo(10000, date(2025, 10, 1)), "acme", "2026-02")

    assert invoice["subtotal_fen"] == 10000
    assert invoice["lines"] == [{"label": "Pro", "amount_fen": 10000}]


def test_half_month_rounds_half_to_even():
    invoice = get_invoice(_repo(9999, date(2026, 2, 15)), "acme", "2026-02")

    assert invoice["lines"][0]["amount_fen"] == 5000
    assert invoice["subtotal_fen"] == 5000


def test_half_fen_goes_to_the_even_neighbour():
    # 12345 * 14 / 28 = 6172.5 -> 6172 (even), not 6173.
    invoice = get_invoice(_repo(12345, date(2026, 2, 1), date(2026, 2, 14)), "acme", "2026-02")

    assert invoice["subtotal_fen"] == 6172

    # 12347 * 14 / 28 = 6173.5 -> 6174.
    other = get_invoice(_repo(12347, date(2026, 2, 1), date(2026, 2, 14)), "acme", "2026-02")
    assert other["subtotal_fen"] == 6174


def test_leap_february_uses_29_days():
    # 9999 * 15 / 29 = 5171.896... -> 5172.
    invoice = get_invoice(_repo(9999, date(2028, 2, 15)), "acme", "2028-02")

    assert invoice["subtotal_fen"] == 5172


def test_end_date_is_inclusive():
    invoice = get_invoice(
        _repo(9999, date(2026, 2, 10), date(2026, 2, 10)), "acme", "2026-02"
    )

    assert invoice["subtotal_fen"] == 357


def test_subscription_overlapping_the_month_start():
    # Active 2026-02-01 .. 2026-02-10 -> 10 of 28 days.
    invoice = get_invoice(
        _repo(10000, date(2026, 1, 10), date(2026, 2, 10)), "acme", "2026-02"
    )

    assert invoice["subtotal_fen"] == 3571


def test_month_without_coverage_is_empty():
    invoice = get_invoice(_repo(10000, date(2026, 3, 1)), "acme", "2026-02")

    assert invoice["lines"] == []
    assert invoice["subtotal_fen"] == 0
    assert invoice["total_fen"] == 0


def test_addons_are_prorated_and_ordered_by_code():
    repo = _repo(
        10000,
        date(2025, 10, 1),
        addons=(
            Addon(code="z", monthly_fee_fen=3000, start_date=date(2025, 10, 1)),
            Addon(code="a", monthly_fee_fen=2000, start_date=date(2025, 10, 1)),
        ),
    )

    invoice = get_invoice(repo, "acme", "2026-02")

    assert [line["label"] for line in invoice["lines"]] == ["Pro", "a", "z"]
    assert invoice["subtotal_fen"] == 15000


def test_addon_starting_mid_month():
    repo = _repo(
        10000,
        date(2025, 10, 1),
        addons=(Addon(code="a", monthly_fee_fen=5000, start_date=date(2026, 2, 15)),),
    )

    assert get_invoice(repo, "acme", "2026-02")["subtotal_fen"] == 12500


def test_addon_ending_mid_month():
    repo = _repo(
        10000,
        date(2025, 10, 1),
        addons=(
            Addon(
                code="a",
                monthly_fee_fen=5000,
                start_date=date(2025, 10, 1),
                end_date=date(2026, 2, 15),
            ),
        ),
    )

    # 5000 * 15 / 28 = 2678.57 -> 2679.
    assert get_invoice(repo, "acme", "2026-02")["subtotal_fen"] == 12679


def test_addon_is_not_billed_before_the_subscription():
    repo = _repo(
        10000,
        date(2026, 2, 20),
        addons=(Addon(code="a", monthly_fee_fen=2800, start_date=date(2026, 1, 1)),),
    )

    # Plan: 9 days of 28. Addon: only the same 9 days, not the whole month.
    assert get_invoice(repo, "acme", "2026-02")["subtotal_fen"] == 4114


def test_per_invoice_rounds_once_across_lines(monkeypatch):
    monkeypatch.delenv("LEDGER_ROUNDING_MODE", raising=False)
    repo = _repo(
        1,
        date(2026, 2, 15),
        addons=(Addon(code="a", monthly_fee_fen=1, start_date=date(2026, 2, 15)),),
    )

    invoice = get_invoice(repo, "acme", "2026-02")

    assert invoice["rounding"] == "per_invoice"
    assert invoice["subtotal_fen"] == 1, "0.5 + 0.5 rounds to 1 once"
    assert [line["amount_fen"] for line in invoice["lines"]] == [0, 0]


def test_explicit_per_line_mode_is_honoured():
    repo = _repo(
        1,
        date(2026, 2, 15),
        addons=(Addon(code="a", monthly_fee_fen=1, start_date=date(2026, 2, 15)),),
    )

    invoice = get_invoice(repo, "acme", "2026-02", Settings(rounding_mode=PER_LINE))

    assert invoice["rounding"] == "per_line"
    assert invoice["subtotal_fen"] == 0, "0.5 + 0.5 rounds to 0 + 0 per line"


def test_environment_override():
    assert from_env({}).rounding_mode == PER_INVOICE
    assert from_env({"LEDGER_ROUNDING_MODE": PER_LINE}).rounding_mode == PER_LINE
    with pytest.raises(ValueError):
        from_env({"LEDGER_ROUNDING_MODE": "per_cent"})


def test_credit_is_applied_after_rounding_and_carried():
    repo = _repo(
        1,
        date(2026, 2, 15),
        addons=(Addon(code="a", monthly_fee_fen=1, start_date=date(2026, 2, 15)),),
        credits={"acme": 5},
    )

    invoice = get_invoice(repo, "acme", "2026-02")

    assert invoice["subtotal_fen"] == 1
    assert invoice["credit_applied_fen"] == 1
    assert invoice["credit_carried_fen"] == 4
    assert invoice["total_fen"] == 0
    assert repo.credit_balance("acme") == 4


def test_credit_carries_into_the_next_month():
    repo = _repo(10000, date(2025, 10, 1), credits={"acme": 2500})

    first = get_invoice(repo, "acme", "2026-02")
    second = get_invoice(repo, "acme", "2026-03")

    assert first["credit_applied_fen"] == 2500
    assert first["total_fen"] == 7500
    assert second["credit_applied_fen"] == 0
    assert second["total_fen"] == 10000


def test_unused_credit_survives_a_zero_month():
    repo = _repo(10000, date(2026, 3, 1), credits={"acme": 700})

    invoice = get_invoice(repo, "acme", "2026-02")

    assert invoice["credit_applied_fen"] == 0
    assert invoice["credit_carried_fen"] == 700
    assert invoice["total_fen"] == 0


def test_bad_inputs():
    repo = _repo(10000, date(2025, 10, 1))

    with pytest.raises(KeyError):
        get_invoice(repo, "nobody", "2026-02")
    with pytest.raises(ValueError):
        get_invoice(repo, "acme", "2026-13")
    with pytest.raises(ValueError):
        get_invoice(repo, "acme", "2026-2")
