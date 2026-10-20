"""Hidden checks for h7-event-rollup. Never copied into the agent's repo."""

import json
import random
import subprocess
import sys

import pytest
from rollup import build_rollup

MINUTE = 60 * 1000
GAP = 30 * 60 * 1000


def _event(user: str, ts_ms: int, tenant: str = "acme") -> dict:
    return {"tenant": tenant, "user": user, "ts_ms": ts_ms}


def test_empty_input():
    assert build_rollup([]) == {}


def test_single_event():
    assert build_rollup([_event("u1", 0)]) == {
        "acme": {0: {"count": 1, "unique_sessions": 1, "max_gap_ms": 0}}
    }


def test_max_gap_inside_one_minute():
    rollup = build_rollup([_event("u1", 0), _event("u1", 1000), _event("u1", 2000)])

    assert rollup["acme"][0] == {"count": 3, "unique_sessions": 1, "max_gap_ms": 1000}


def test_identical_timestamps():
    events = [_event("u1", 5000) for _ in range(5)]

    assert build_rollup(events)["acme"][0] == {
        "count": 5,
        "unique_sessions": 1,
        "max_gap_ms": 0,
    }


def test_gap_boundary_still_one_session():
    rollup = build_rollup([_event("u1", 0), _event("u1", GAP)])

    assert rollup == {
        "acme": {
            0: {"count": 1, "unique_sessions": 1, "max_gap_ms": 0},
            30: {"count": 1, "unique_sessions": 1, "max_gap_ms": 0},
        }
    }


def test_gap_just_over_the_boundary_is_a_new_session():
    rollup = build_rollup([_event("u1", 0), _event("u1", GAP + 1)])

    assert rollup["acme"][0]["unique_sessions"] == 1
    assert rollup["acme"][30]["unique_sessions"] == 1


def test_sessions_chain_transitively():
    # 0 / 25 min / 50 min: first and last are 50 minutes apart but one session.
    rollup = build_rollup([_event("u1", 0), _event("u1", 25 * MINUTE), _event("u1", 50 * MINUTE)])

    assert set(rollup["acme"]) == {0, 25, 50}
    assert all(bucket["unique_sessions"] == 1 for bucket in rollup["acme"].values())


def test_gaps_across_minutes_are_not_attributed_to_a_bucket():
    rollup = build_rollup(
        [_event("u1", 0), _event("u1", 10_000), _event("u1", GAP)]
    )

    assert rollup["acme"][0]["max_gap_ms"] == 10_000
    assert rollup["acme"][30]["max_gap_ms"] == 0


def test_two_users_are_two_sessions():
    rollup = build_rollup([_event("u1", 0), _event("u2", 10)])

    assert rollup["acme"][0] == {"count": 2, "unique_sessions": 2, "max_gap_ms": 0}


def test_users_have_independent_sessions():
    rollup = build_rollup(
        [
            _event("u1", 0),
            _event("u2", 500),
            _event("u1", 1000),
        ]
    )

    assert rollup["acme"][0] == {"count": 3, "unique_sessions": 2, "max_gap_ms": 1000}


def test_tenants_are_separate():
    rollup = build_rollup([_event("u1", 0, tenant="a"), _event("u1", 0, tenant="b")])

    assert set(rollup) == {"a", "b"}
    assert rollup["a"][0]["count"] == 1


def test_unsorted_input():
    events = [_event("u1", 50 * MINUTE), _event("u1", 0), _event("u1", 25 * MINUTE)]

    assert set(build_rollup(events)["acme"]) == {0, 25, 50}


def test_minute_boundary_keeps_one_session():
    rollup = build_rollup([_event("u1", 59_999), _event("u1", 60_000)])

    assert rollup["acme"][0]["count"] == 1
    assert rollup["acme"][1]["count"] == 1
    assert rollup["acme"][0]["unique_sessions"] == 1
    assert rollup["acme"][1]["unique_sessions"] == 1


def test_old_events_do_not_join_new_sessions():
    rollup = build_rollup([_event("u1", 0), _event("u1", 100 * MINUTE)])

    assert rollup["acme"][0]["unique_sessions"] == 1
    assert rollup["acme"][100]["unique_sessions"] == 1


def test_input_validation():
    with pytest.raises(KeyError):
        build_rollup([{"tenant": "acme", "user": "u1"}])
    with pytest.raises(ValueError):
        build_rollup([_event("u1", -1)])
    with pytest.raises(TypeError):
        build_rollup([_event("u1", 1.5)])
    with pytest.raises(TypeError):
        build_rollup(["not a dict"])


class _Reference:
    """A deliberately simple, independent implementation of the README."""

    def __init__(self, events: list[dict]) -> None:
        self.events = events

    def run(self) -> dict:
        groups: dict[tuple, list[tuple[int, int]]] = {}
        for index, event in enumerate(self.events):
            groups.setdefault((event["tenant"], event["user"]), []).append((event["ts_ms"], index))

        session_of: dict[int, str] = {}
        for (tenant, user), items in groups.items():
            items.sort()
            start = None
            previous = None
            for ts_ms, index in items:
                if previous is None or ts_ms - previous > GAP:
                    start = ts_ms
                session_of[index] = f"{tenant}:{user}:{start}"
                previous = ts_ms

        buckets: dict[str, dict[int, dict]] = {}
        for index, event in enumerate(self.events):
            minute = event["ts_ms"] // MINUTE
            bucket = buckets.setdefault(event["tenant"], {}).setdefault(
                minute, {"count": 0, "sessions": set(), "max_gap_ms": 0}
            )
            bucket["count"] += 1
            bucket["sessions"].add(session_of[index])

        for (tenant, user), items in groups.items():
            items.sort()
            for (earlier, _), (later, _) in zip(items, items[1:]):
                if earlier // MINUTE != later // MINUTE:
                    continue
                bucket = buckets[tenant][earlier // MINUTE]
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


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
def test_random_inputs_match_the_reference(seed: int):
    rng = random.Random(seed)
    events = []
    for _ in range(rng.randint(1, 70)):
        events.append(
            {
                "tenant": rng.choice(["acme", "globex"]),
                "user": f"u{rng.randrange(4)}",
                "ts_ms": rng.randrange(0, 20) * MINUTE + rng.randrange(0, 3) * GAP,
            }
        )
    rng.shuffle(events)

    assert build_rollup(events) == _Reference(events).run()


def test_scale_120k_events_finishes_well_inside_budget():
    """The production replay is ~120k events; the naive pairwise stitch times out."""
    events = [
        {"tenant": "acme", "user": f"u{index % 2}", "ts_ms": index * 10}
        for index in range(120_000)
    ]
    script = (
        "import json, sys\n"
        "from rollup import build_rollup\n"
        "result = build_rollup(json.load(sys.stdin))\n"
        "print(json.dumps({'total': sum(b['count'] for m in result.values() for b in m.values())}))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(events),
        capture_output=True,
        text=True,
        timeout=9,
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert json.loads(completed.stdout)["total"] == 120_000
