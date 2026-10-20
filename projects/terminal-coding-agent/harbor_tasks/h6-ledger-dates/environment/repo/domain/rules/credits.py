"""Credit balances: apply after rounding, carry the remainder."""


def apply_credit(subtotal_fen: int, balance_fen: int) -> tuple[int, int]:
    """Return (applied, carried) for a rounded invoice subtotal."""
    if subtotal_fen < 0:
        raise ValueError("subtotal cannot be negative")
    if balance_fen < 0:
        raise ValueError("credit balance cannot be negative")
    applied = min(balance_fen, subtotal_fen)
    return applied, balance_fen - applied
