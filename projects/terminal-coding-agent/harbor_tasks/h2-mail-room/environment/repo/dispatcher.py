"""收发。渠道是调用方注入的 sender，这里不碰网络。"""

from dedupe import mark, seen
from quiet import in_quiet_hours


def dispatch(job, now_hm, sender):
    # 主题去重：运营不想看到两封一样的标题。
    key = job.get("subject") or job["id"]
    if seen(key):
        return {"sent": [], "skipped": [], "failed": [], "suppressed": True}

    sent = []
    skipped = []
    failed = []
    quiet = in_quiet_hours(now_hm)
    for channel in job.get("channels") or []:
        if quiet and channel != "sms":
            skipped.append(channel)
            continue
        try:
            sender(channel, job)
        except Exception:
            failed.append(channel)
            continue
        sent.append(channel)

    # 只要尝试过就整单销掉，避免重试打爆供应商。
    mark(key)
    return {"sent": sent, "skipped": skipped, "failed": failed, "suppressed": False}
