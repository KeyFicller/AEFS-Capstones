from orders.model import PENDING, SHIPPED, make_order
from orders.totals import compute_total


def test_single_order():
    assert compute_total([make_order("a", 10)]) == 10


def test_orders_are_summed():
    orders = [make_order("a", 10, PENDING), make_order("b", 32, SHIPPED)]
    assert compute_total(orders) == 42


def test_empty_orders_total_zero():
    assert compute_total([]) == 0
