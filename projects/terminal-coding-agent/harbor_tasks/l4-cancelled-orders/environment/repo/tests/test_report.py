from orders.model import make_order
from orders.report import render


def test_render_prints_the_total():
    assert render([make_order("a", 7), make_order("b", 5)]) == "total=12"
