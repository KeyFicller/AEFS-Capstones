"""Visible regression suite: small samples only."""

from rollup import build_rollup

MINUTE = 60 * 1000
GAP = 30 * 60 * 1000


def _event(user: str, ts_ms: int, tenant: str = "acme") -> dict:
    return {"tenant": tenant, "user": user, "ts_ms": ts_ms}


def test_single_event():
    rollup = build_rollup([_event("u1", 1000)])

    assert rollup == {"acme": {0: {"count": 1, "unique_sessions": 1, "max_gap_ms": 0}}}


def test_two_events_in_one_minute():
    rollup = build_rollup([_event("u1", 1000), _event("u1", 5000)])

    assert rollup == {"acme": {0: {"count": 2, "unique_sessions": 1, "max_gap_ms": 4000}}}


def test_gap_boundary_keeps_one_session():
    rollup = build_rollup([_event("u1", 0), _event("u1", GAP)])

    assert rollup == {
        "acme": {
            0: {"count": 1, "unique_sessions": 1, "max_gap_ms": 0},
            30: {"count": 1, "unique_sessions": 1, "max_gap_ms": 0},
        }
    }


def test_events_are_sorted_before_stitching():
    rollup = build_rollup([_event("u1", 2 * MINUTE), _event("u1", 0)])

    assert rollup["acme"][0]["count"] == 1
    assert rollup["acme"][2]["count"] == 1


def test_two_users_count_as_two_sessions():
    rollup = build_rollup([_event("u1", 0), _event("u2", 10)])

    assert rollup["acme"][0] == {"count": 2, "unique_sessions": 2, "max_gap_ms": 0}
