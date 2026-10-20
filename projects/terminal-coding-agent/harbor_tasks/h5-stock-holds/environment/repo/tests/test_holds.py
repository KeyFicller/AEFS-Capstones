"""Visible regression suite: happy paths only."""

from engine import Engine


def test_reserve_returns_an_active_hold():
    engine = Engine({"apple": 10})

    hold = engine.reserve("k1", "apple", 3, ttl=10, now=0)

    assert hold.id == "h1"
    assert hold.qty == 3
    assert hold.expires_at == 10
    assert hold.state == "active"


def test_reserve_refuses_more_than_stock():
    engine = Engine({"apple": 2})

    assert engine.reserve("k1", "apple", 3, ttl=10, now=0) is None
    assert engine.available("apple", 0) == 2


def test_confirm_marks_the_hold_confirmed():
    engine = Engine({"apple": 10})
    hold = engine.reserve("k1", "apple", 3, ttl=10, now=0)

    confirmed = engine.confirm(hold.id, now=1)

    assert confirmed.state == "confirmed"


def test_release_returns_the_quantity():
    engine = Engine({"apple": 10})
    hold = engine.reserve("k1", "apple", 3, ttl=10, now=0)
    assert engine.available("apple", 0) == 7

    engine.release(hold.id, now=1)

    assert engine.available("apple", 1) == 10
