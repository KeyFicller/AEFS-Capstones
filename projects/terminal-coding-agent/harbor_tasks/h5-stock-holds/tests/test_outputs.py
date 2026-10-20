"""Hidden checks for h5-stock-holds. Never copied into the agent's repo."""

import random

import pytest
from engine import Engine
from models import ConflictError, StateError

STOCK = {"apple": 10, "pear": 5}


def _engine() -> Engine:
    return Engine(dict(STOCK))


def _states(rows: list[dict]) -> list[tuple]:
    return [(row["id"], row["key"], row["sku"], row["qty"], row["expires_at"], row["state"]) for row in rows]


def test_reserve_occupies_stock():
    engine = _engine()

    hold = engine.reserve("k1", "apple", 3, ttl=10, now=0)

    assert (hold.id, hold.expires_at, hold.state) == ("h1", 10, "active")
    assert engine.available("apple", 0) == 7


def test_reserve_returns_none_without_touching_state():
    engine = _engine()
    engine.reserve("k1", "apple", 8, ttl=10, now=0)

    assert engine.reserve("k2", "apple", 3, ttl=10, now=0) is None
    assert engine.available("apple", 0) == 2
    assert len(engine.snapshot(0)) == 1


def test_expiry_is_inclusive_and_lazy():
    engine = _engine()
    engine.reserve("k1", "apple", 3, ttl=5, now=0)

    assert engine.available("apple", 4) == 7
    assert engine.available("apple", 5) == 10, "now >= expires_at means expired"


def test_reserve_reclaims_expired_stock():
    """A reservation must expire lazily before the stock check, not after a read."""
    engine = _engine()
    engine.reserve("k1", "apple", 8, ttl=5, now=0)

    assert engine.reserve("k2", "apple", 8, ttl=5, now=6) is not None
    assert engine.available("apple", 6) == 2


def test_idempotent_replay_does_not_consume_twice():
    engine = _engine()
    first = engine.reserve("k1", "apple", 3, ttl=10, now=0)

    second = engine.reserve("k1", "apple", 3, ttl=10, now=1)

    assert second.id == first.id
    assert engine.available("apple", 1) == 7


def test_idempotent_replay_returns_the_confirmed_hold():
    engine = _engine()
    first = engine.reserve("k1", "apple", 3, ttl=10, now=0)
    engine.confirm(first.id, now=1)

    replay = engine.reserve("k1", "apple", 3, ttl=10, now=2)

    assert replay.id == first.id
    assert replay.state == "confirmed"
    assert engine.available("apple", 2) == 7


def test_same_key_with_a_different_quantity_conflicts():
    engine = _engine()
    engine.reserve("k1", "apple", 3, ttl=10, now=0)

    with pytest.raises(ConflictError):
        engine.reserve("k1", "apple", 4, ttl=10, now=1)


def test_same_key_with_a_different_sku_conflicts():
    engine = _engine()
    engine.reserve("k1", "apple", 3, ttl=10, now=0)

    with pytest.raises(ConflictError):
        engine.reserve("k1", "pear", 3, ttl=10, now=1)


def test_confirm_moves_the_quantity_to_consumed():
    engine = _engine()
    hold = engine.reserve("k1", "apple", 3, ttl=10, now=0)

    engine.confirm(hold.id, now=1)

    assert engine.available("apple", 1) == 7, "the same stock only changed owner"
    assert [row["state"] for row in engine.snapshot(1)] == ["confirmed"]


def test_confirm_is_idempotent():
    engine = _engine()
    hold = engine.reserve("k1", "apple", 3, ttl=10, now=0)
    engine.confirm(hold.id, now=1)

    engine.confirm(hold.id, now=2)

    assert engine.available("apple", 2) == 7


def test_cannot_confirm_a_released_hold():
    engine = _engine()
    hold = engine.reserve("k1", "apple", 3, ttl=10, now=0)
    engine.release(hold.id, now=1)

    with pytest.raises(StateError):
        engine.confirm(hold.id, now=2)


def test_cannot_confirm_an_expired_hold():
    engine = _engine()
    hold = engine.reserve("k1", "apple", 3, ttl=5, now=0)

    with pytest.raises(StateError):
        engine.confirm(hold.id, now=5)


def test_release_returns_stock_and_is_idempotent():
    engine = _engine()
    hold = engine.reserve("k1", "apple", 3, ttl=10, now=0)

    engine.release(hold.id, now=1)
    engine.release(hold.id, now=2)

    assert engine.available("apple", 2) == 10


def test_cannot_release_a_confirmed_hold():
    engine = _engine()
    hold = engine.reserve("k1", "apple", 3, ttl=10, now=0)
    engine.confirm(hold.id, now=1)

    with pytest.raises(StateError):
        engine.release(hold.id, now=2)


def test_snapshot_hides_expired_and_released_holds():
    engine = _engine()
    short = engine.reserve("k1", "apple", 1, ttl=2, now=0)
    released = engine.reserve("k2", "apple", 1, ttl=10, now=0)
    kept = engine.reserve("k3", "pear", 2, ttl=10, now=0)
    engine.release(released.id, now=1)

    rows = engine.snapshot(2)

    assert [row["id"] for row in rows] == [kept.id], f"{short.id} expired, {released.id} released"


def test_snapshot_keeps_creation_order():
    engine = _engine()
    ids = [engine.reserve(f"k{i}", "apple", 1, ttl=10, now=0).id for i in range(3)]

    assert [row["id"] for row in engine.snapshot(0)] == ids


def test_inputs_are_validated():
    engine = _engine()

    with pytest.raises(ValueError):
        engine.reserve("k1", "apple", 0, ttl=1, now=0)
    with pytest.raises(ValueError):
        engine.reserve("k1", "apple", 1, ttl=0, now=0)
    with pytest.raises(KeyError):
        engine.reserve("k1", "durian", 1, ttl=1, now=0)
    with pytest.raises(KeyError):
        engine.confirm("h999", now=0)
    with pytest.raises(KeyError):
        engine.release("h999", now=0)
    with pytest.raises(KeyError):
        engine.available("durian", 0)


class _Ref:
    """Recompute the documented semantics from scratch: an oracle for the fuzz check."""

    def __init__(self, stock: dict) -> None:
        self.stock = dict(stock)
        self.holds: dict[str, dict] = {}
        self.requests: dict[str, tuple] = {}
        self.next_id = 1

    def _expire(self, now: int) -> None:
        for hold in self.holds.values():
            if hold["state"] == "active" and now >= hold["expires_at"]:
                hold["state"] = "expired"

    def _held(self, sku: str) -> int:
        return sum(h["qty"] for h in self.holds.values() if h["sku"] == sku and h["state"] == "active")

    def _consumed(self, sku: str) -> int:
        return sum(
            h["qty"] for h in self.holds.values() if h["sku"] == sku and h["state"] == "confirmed"
        )

    def available(self, sku: str, now: int) -> int:
        if sku not in self.stock:
            raise KeyError(sku)
        self._expire(now)
        return self.stock[sku] - self._consumed(sku) - self._held(sku)

    def reserve(self, key: str, sku: str, qty: int, ttl: int, now: int):
        if qty <= 0 or ttl <= 0:
            raise ValueError("qty and ttl must be positive")
        if sku not in self.stock:
            raise KeyError(sku)
        self._expire(now)
        if key in self.requests:
            first_sku, first_qty, hold_id = self.requests[key]
            if (first_sku, first_qty) != (sku, qty):
                raise ConflictError(key)
            return self.holds[hold_id]
        if qty > self.stock[sku] - self._consumed(sku) - self._held(sku):
            return None
        hold_id = f"h{self.next_id}"
        self.next_id += 1
        self.holds[hold_id] = {
            "id": hold_id,
            "key": key,
            "sku": sku,
            "qty": qty,
            "expires_at": now + ttl,
            "state": "active",
        }
        self.requests[key] = (sku, qty, hold_id)
        return self.holds[hold_id]

    def confirm(self, hold_id: str, now: int):
        self._expire(now)
        hold = self.holds[hold_id]
        if hold["state"] == "confirmed":
            return hold
        if hold["state"] in ("released", "expired"):
            raise StateError(hold["state"])
        hold["state"] = "confirmed"
        return hold

    def release(self, hold_id: str, now: int) -> None:
        self._expire(now)
        hold = self.holds[hold_id]
        if hold["state"] in ("released", "expired"):
            return
        if hold["state"] == "confirmed":
            raise StateError("confirmed")
        hold["state"] = "released"

    def snapshot(self, now: int) -> list[dict]:
        self._expire(now)
        return [
            {
                "id": h["id"],
                "key": h["key"],
                "sku": h["sku"],
                "qty": h["qty"],
                "expires_at": h["expires_at"],
                "state": h["state"],
            }
            for h in self.holds.values()
            if h["state"] in ("active", "confirmed")
        ]


def _outcome(call):
    try:
        return ("ok", call())
    except Exception as exc:  # noqa: BLE001 - the exception type is the assertion
        return ("err", type(exc).__name__)


def _hold_key(value) -> tuple:
    if isinstance(value, dict):
        return tuple(value.items())
    if hasattr(value, "as_dict"):
        return tuple(value.as_dict().items())
    return value


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_random_operation_sequences_match_the_documented_semantics(seed: int) -> None:
    rng = random.Random(seed)
    engine = _engine()
    ref = _Ref(STOCK)
    known: list[str] = []

    for step in range(400):
        choose = rng.random()
        now = step
        if choose < 0.42:
            key = f"k{rng.randrange(6)}"
            sku = rng.choice(["apple", "pear"])
            qty = rng.randint(1, 4)
            ttl = rng.randint(1, 8)
            args = (key, sku, qty, ttl, now)
            got = _outcome(lambda: engine.reserve(*args))
            want = _outcome(lambda: ref.reserve(*args))
            if want[0] == "ok" and want[1] is not None:
                known.append(want[1]["id"] if isinstance(want[1], dict) else want[1].id)
            assert (got[0], _hold_key(got[1]) if got[0] == "ok" else got[1]) == (
                want[0],
                _hold_key(want[1]) if want[0] == "ok" else want[1],
            ), f"reserve{args} at step {step}"
        elif choose < 0.58:
            hold_id = rng.choice(known + ["h999"]) if known else "h999"
            got = _outcome(lambda: engine.confirm(hold_id, now))
            want = _outcome(lambda: ref.confirm(hold_id, now))
            assert (got[0], _hold_key(got[1]) if got[0] == "ok" else got[1]) == (
                want[0],
                _hold_key(want[1]) if want[0] == "ok" else want[1],
            ), f"confirm({hold_id}) at step {step}"
        elif choose < 0.72:
            hold_id = rng.choice(known + ["h999"]) if known else "h999"
            assert _outcome(lambda: engine.release(hold_id, now)) == _outcome(
                lambda: ref.release(hold_id, now)
            ), f"release({hold_id}) at step {step}"
        elif choose < 0.86:
            sku = rng.choice(["apple", "pear", "durian"])
            assert _outcome(lambda: engine.available(sku, now)) == _outcome(
                lambda: ref.available(sku, now)
            ), f"available({sku}) at step {step}"
        else:
            assert _states(engine.snapshot(now)) == _states(ref.snapshot(now)), (
                f"snapshot at step {step}"
            )
