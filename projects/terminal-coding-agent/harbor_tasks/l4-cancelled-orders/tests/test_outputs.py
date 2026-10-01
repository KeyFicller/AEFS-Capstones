"""Hidden checks for l4-cancelled-orders. Never copied into the agent's repo."""

from orders.model import CANCELLED, PENDING, SHIPPED, make_order
from orders.report import render
from orders.totals import compute_total


def test_cancelled_orders_are_excluded():
    orders = [make_order("a", 10), make_order("b", 99, CANCELLED)]
    assert compute_total(orders) == 10


def test_all_cancelled_totals_zero():
    orders = [make_order("a", 99, CANCELLED), make_order("b", 1, CANCELLED)]
    assert compute_total(orders) == 0


def test_uncancelled_orders_are_still_summed():
    orders = [
        make_order("a", 5, PENDING),
        make_order("b", 7, SHIPPED),
        make_order("c", 100, CANCELLED),
    ]
    assert compute_total(orders) == 12


def test_empty_is_zero():
    assert compute_total([]) == 0


def test_report_keeps_printing_the_uncancelled_total():
    orders = [make_order("a", 7), make_order("b", 5, CANCELLED)]
    assert render(orders) == "total=7"
