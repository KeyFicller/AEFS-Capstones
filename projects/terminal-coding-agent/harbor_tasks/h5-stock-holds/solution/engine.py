"""Stock reservation engine: holds, confirmation, release and lazy expiry."""

from models import ConflictError, Hold, StateError

ACTIVE = "active"
CONFIRMED = "confirmed"
RELEASED = "released"
EXPIRED = "expired"


class Engine:
    def __init__(self, stock: dict) -> None:
        self._stock = {sku: int(qty) for sku, qty in stock.items()}
        self._holds: dict[str, Hold] = {}
        self._requests: dict[str, tuple[str, int, str]] = {}
        self._next_id = 1

    def _expire(self, now: int) -> None:
        for hold in self._holds.values():
            if hold.state == ACTIVE and now >= hold.expires_at:
                hold.state = EXPIRED

    def _check_sku(self, sku: str) -> None:
        if sku not in self._stock:
            raise KeyError(sku)

    def _held(self, sku: str) -> int:
        return sum(h.qty for h in self._holds.values() if h.sku == sku and h.state == ACTIVE)

    def _consumed(self, sku: str) -> int:
        return sum(h.qty for h in self._holds.values() if h.sku == sku and h.state == CONFIRMED)

    def _free(self, sku: str) -> int:
        return self._stock[sku] - self._consumed(sku) - self._held(sku)

    def available(self, sku: str, now: int) -> int:
        self._check_sku(sku)
        self._expire(now)
        return self._free(sku)

    def reserve(self, key: str, sku: str, qty: int, ttl: int, now: int) -> Hold | None:
        if qty <= 0:
            raise ValueError("qty must be positive")
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        self._check_sku(sku)
        self._expire(now)

        if key in self._requests:
            first_sku, first_qty, hold_id = self._requests[key]
            if (first_sku, first_qty) != (sku, qty):
                raise ConflictError(f"{key} was used for {first_sku} x{first_qty}")
            return self._holds[hold_id]

        if qty > self._free(sku):
            return None

        hold = Hold(
            id=f"h{self._next_id}",
            key=key,
            sku=sku,
            qty=qty,
            created_at=now,
            expires_at=now + ttl,
        )
        self._next_id += 1
        self._holds[hold.id] = hold
        self._requests[key] = (sku, qty, hold.id)
        return hold

    def confirm(self, hold_id: str, now: int) -> Hold:
        self._expire(now)
        hold = self._holds[hold_id]
        if hold.state == CONFIRMED:
            return hold
        if hold.state in (RELEASED, EXPIRED):
            raise StateError(f"cannot confirm a {hold.state} hold")
        hold.state = CONFIRMED
        return hold

    def release(self, hold_id: str, now: int) -> None:
        self._expire(now)
        hold = self._holds[hold_id]
        if hold.state in (RELEASED, EXPIRED):
            return
        if hold.state == CONFIRMED:
            raise StateError("cannot release a confirmed hold")
        hold.state = RELEASED

    def snapshot(self, now: int) -> list[dict]:
        self._expire(now)
        return [
            hold.as_dict()
            for hold in self._holds.values()
            if hold.state in (ACTIVE, CONFIRMED)
        ]
