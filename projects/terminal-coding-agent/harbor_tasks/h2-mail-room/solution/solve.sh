#!/bin/bash
# Reference solution. Only touches source files, never the visible tests.
set -euo pipefail

cd /app

python3 - <<'PY'
from pathlib import Path

Path("quiet.py").write_text(
    '''"""静默时段 [22:00, 07:00)。"""


def in_quiet_hours(hm):
    return hm >= "22:00" or hm < "07:00"
'''
)

Path("dispatcher.py").write_text(
    '''"""收发。去重粒度是 (user, id, channel)，只记成功。"""

from dedupe import mark, seen
from quiet import in_quiet_hours


def _key(job, channel):
    return (job["user_id"], job["id"], channel)


def dispatch(job, now_hm, sender):
    sent = []
    skipped = []
    failed = []
    quiet = in_quiet_hours(now_hm)
    for channel in job.get("channels") or []:
        if seen(_key(job, channel)):
            continue
        if quiet and channel != "sms":
            skipped.append(channel)
            continue
        try:
            sender(channel, job)
        except Exception:
            failed.append(channel)
            continue
        mark(_key(job, channel))
        sent.append(channel)
    return {"sent": sent, "skipped": skipped, "failed": failed, "suppressed": False}
'''
)
PY

python3 -m pytest tests -q
