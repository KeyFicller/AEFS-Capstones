"""可见回归：7 天试用的第一天收费。"""

from billing import bill


def test_first_charge_is_the_day_after_an_inclusive_trial():
    amount = bill(
        {
            "plan": "basic",
            "status": "active",
            "started": "2024-01-01",
            "trial_days": 7,
        },
        "2024-01-08",
    )

    assert amount == 2900
