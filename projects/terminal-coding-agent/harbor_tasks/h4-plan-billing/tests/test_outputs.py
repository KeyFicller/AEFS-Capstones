"""Hidden checks for h4-plan-billing. Never copied into the agent's repo."""

from billing import bill


def _sub(**overrides):
    sub = {
        "plan": "basic",
        "status": "active",
        "started": "2024-01-01",
        "trial_days": 7,
    }
    sub.update(overrides)
    return sub


def test_last_trial_day_is_free_and_the_next_day_is_not_an_anniversary():
    assert bill(_sub(), "2024-01-07") == 0
    assert bill(_sub(), "2024-01-09") == 0
    assert bill(_sub(), "2024-02-08") == 2900


def test_cancelled_any_case_never_charges():
    assert (
        bill(
            _sub(status="Cancelled", started="2024-01-08", trial_days=0),
            "2024-01-08",
        )
        == 0
    )


def test_active_status_ignores_case_and_pro_price():
    assert (
        bill(
            _sub(plan="pro", status="ACTIVE", started="2024-03-02", trial_days=0),
            "2024-03-02",
        )
        == 9900
    )


def test_unknown_status_charges_nothing():
    assert bill(_sub(status="paused", trial_days=0), "2024-01-01") == 0


def test_anchor_day_falls_on_the_real_month_end():
    sub = _sub(started="2024-01-31", trial_days=0)

    assert bill(sub, "2024-02-28") == 0
    assert bill(sub, "2024-02-29") == 2900
    assert bill(sub, "2024-03-31") == 2900
    assert bill(sub, "2023-12-31") == 0
