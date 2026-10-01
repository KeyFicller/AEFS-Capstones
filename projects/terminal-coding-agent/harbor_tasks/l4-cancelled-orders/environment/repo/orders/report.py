"""One-line totals report."""

from orders.totals import compute_total


def render(orders: list[dict]) -> str:
    """Total line for the given orders."""
    return f"total={compute_total(orders)}"
