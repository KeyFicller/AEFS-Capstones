"""Order totals."""


def compute_total(orders: list[dict]) -> int:
    """Sum the amounts of ``orders``."""
    return sum(order["amount"] for order in orders)
