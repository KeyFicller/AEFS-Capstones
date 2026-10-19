"""收银台回归。只覆盖无运费的百分比券。"""

from checkout import quote


def test_percent_coupon_taxes_discounted_goods():
    quoted = quote(
        {
            "lines": [{"qty": 1, "price_fen": 10000}],
            "coupon": {"type": "percent", "value": 10},
            "shipping_fen": 0,
        }
    )

    assert quoted["subtotal"] == 10000
    assert quoted["discount"] == 1000
    assert quoted["tax"] == 540
    assert quoted["total"] == 9540
