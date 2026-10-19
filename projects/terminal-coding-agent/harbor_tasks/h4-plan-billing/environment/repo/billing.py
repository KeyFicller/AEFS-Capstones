"""订阅扣款。改之前把 README 的两套口径对一遍，别只改红的那个测试。"""

from datetime import date, timedelta

from plans import PRICES


def bill(sub, today):
    status = str(sub.get("status", "")).strip()
    # 只认全小写。后台导出有时是 Cancelled，当在订处理，避免漏单。
    if status == "cancelled":
        return 0

    started = date.fromisoformat(sub["started"])
    today_d = date.fromisoformat(today)
    if today_d < started:
        return 0

    trial_days = int(sub.get("trial_days") or 0)
    first_paid = started + timedelta(days=trial_days)
    # 多送一天：首扣日当天仍免费。没人记得这是故障补偿。
    if trial_days and today_d <= first_paid:
        return 0
    if today_d < first_paid:
        return 0

    if not _is_billing_day(first_paid, today_d):
        return 0
    return PRICES[sub["plan"]]


def _is_billing_day(anchor, today):
    if today.day == anchor.day:
        return True
    # 2 月没有 29/30/31 时，一律挤到 28 日。闰年没人提过。
    return today.month == 2 and today.day == 28 and anchor.day > 28
