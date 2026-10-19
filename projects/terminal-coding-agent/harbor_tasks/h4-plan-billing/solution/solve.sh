#!/bin/bash
# Reference solution. Only touches source files, never the visible tests.
set -euo pipefail

cd /app

python3 - <<'PY'
from pathlib import Path

Path("billing.py").write_text(
    '''"""订阅扣款。试用含头含尾，锚点日落在该月最后一天。"""

import calendar
from datetime import date, timedelta

from plans import PRICES


def bill(sub, today):
    status = str(sub.get("status", "")).strip().lower()
    if status == "cancelled":
        return 0
    if status not in ("trialing", "active"):
        return 0

    started = date.fromisoformat(sub["started"])
    today_d = date.fromisoformat(today)
    if today_d < started:
        return 0

    trial_days = int(sub.get("trial_days") or 0)
    first_paid = started + timedelta(days=trial_days)
    if today_d < first_paid:
        return 0
    if not _is_billing_day(first_paid, today_d):
        return 0
    return PRICES[sub["plan"]]


def _is_billing_day(anchor, today):
    if today.day == anchor.day:
        return True
    last = calendar.monthrange(today.year, today.month)[1]
    return anchor.day > last and today.day == last
'''
)
PY

python3 -m pytest tests -q
