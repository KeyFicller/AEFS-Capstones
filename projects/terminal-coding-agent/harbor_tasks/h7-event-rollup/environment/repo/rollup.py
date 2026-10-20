"""Per-tenant, per-minute event rollups."""

from sessionize import stitch

MINUTE_MS = 60 * 1000


def _validate(events: list[dict]) -> None:
    for event in events:
        if not isinstance(event, dict):
            raise TypeError("every event must be a dict")
        tenant = event["tenant"]
        user = event["user"]
        ts_ms = event["ts_ms"]
        if not isinstance(tenant, str) or not isinstance(user, str):
            raise TypeError("tenant and user must be strings")
        if not isinstance(ts_ms, int) or isinstance(ts_ms, bool):
            raise TypeError("ts_ms must be an int")
        if ts_ms < 0:
            raise ValueError("ts_ms cannot be negative")


def build_rollup(events: list[dict]) -> dict:
    """Aggregate events into {tenant: {minute: {count, unique_sessions, max_gap_ms}}}."""
    if not isinstance(events, list):
        raise TypeError("events must be a list")
    _validate(events)
    if not events:
        return {}

    assignment = stitch(events)

    buckets: dict[str, dict[int, dict]] = {}
    timelines: dict[str, list[tuple[int, int]]] = {}
    session_tenant: dict[str, str] = {}

    for index, event in enumerate(events):
        session_id = assignment[index]
        session_tenant[session_id] = event["tenant"]
        minute = event["ts_ms"] // MINUTE_MS
        bucket = buckets.setdefault(event["tenant"], {}).setdefault(
            minute, {"count": 0, "sessions": set(), "max_gap_ms": 0}
        )
        bucket["count"] += 1
        bucket["sessions"].add(session_id)
        timelines.setdefault(session_id, []).append((event["ts_ms"], index))

    for session_id, items in timelines.items():
        items.sort()
        tenant = session_tenant[session_id]
        for (earlier, _), (later, _) in zip(items, items[1:]):
            if earlier // MINUTE_MS != later // MINUTE_MS:
                continue
            bucket = buckets[tenant][earlier // MINUTE_MS]
            bucket["max_gap_ms"] = max(bucket["max_gap_ms"], later - earlier)

    return {
        tenant: {
            minute: {
                "count": bucket["count"],
                "unique_sessions": len(bucket["sessions"]),
                "max_gap_ms": bucket["max_gap_ms"],
            }
            for minute, bucket in minutes.items()
        }
        for tenant, minutes in buckets.items()
    }
