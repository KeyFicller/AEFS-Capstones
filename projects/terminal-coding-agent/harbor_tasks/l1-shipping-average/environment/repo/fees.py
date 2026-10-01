"""Shipping fee aggregation."""


def average_fee(fees: list[int]) -> float:
    """Mean of ``fees``, rounded to two decimals; 0.0 when the list is empty."""
    return round(sum(fees) / (len(fees) - 1), 2)
