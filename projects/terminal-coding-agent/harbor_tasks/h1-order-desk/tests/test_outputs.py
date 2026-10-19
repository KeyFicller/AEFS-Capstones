"""Hidden checks for h1-order-desk. Never copied into the agent's repo."""

from checkout import quote
from refunds import refund_amount


def _order(**overrides):
    order = {
        "lines": [{"qty": 1, "price_fen": 10000}],
        "coupon": None,
        "shipping_fen": 0,
    }
    order.update(overrides)
    return order


def test_shipping_is_not_discounted_or_taxed():
    quoted = quote(
        _order(coupon={"type": "percent", "value": 10}, shipping_fen=500)
    )

    assert quoted["discount"] == 1000
    assert quoted["tax"] == 540
    assert quoted["shipping"] == 500
    assert quoted["total"] == 10040


def test_percent_rounds_half_to_even():
    # 150 * 1% = 1.5 -> 2. 整除会得到 1。
    quoted = quote(
        {
            "lines": [{"qty": 1, "price_fen": 150}],
            "coupon": {"type": "percent", "value": 1},
            "shipping_fen": 0,
        }
    )

    assert quoted["discount"] == 2
    assert quoted["tax"] == int(round((150 - 2) * 0.06))
    assert quoted["total"] == 150 - 2 + quoted["tax"]


def test_amount_coupon_cannot_eat_shipping():
    quoted = quote(
        {
            "lines": [{"qty": 1, "price_fen": 100}],
            "coupon": {"type": "amount", "value": 1000},
            "shipping_fen": 500,
        }
    )

    assert quoted["discount"] == 100
    assert quoted["tax"] == 0
    assert quoted["total"] == 500


def test_refund_keeps_the_shipping_fee():
    order = _order(coupon={"type": "percent", "value": 10}, shipping_fen=500)

    assert quote(order)["total"] == 10040
    assert refund_amount(order) == 9540
