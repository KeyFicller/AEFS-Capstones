"""Order records and the statuses they can carry."""

CANCELLED = "cancelled"
PENDING = "pending"
SHIPPED = "shipped"


def make_order(order_id: str, amount: int, status: str = PENDING) -> dict:
    """Build one order record."""
    return {"order_id": order_id, "amount": amount, "status": status}
