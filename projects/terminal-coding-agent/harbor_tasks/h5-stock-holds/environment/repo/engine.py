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
        self._by_key: dict[str, Hold] = {}
        self._active = {sku: 0 for sku in self._stock}
        self._consumed = {sku: 0 for sku in self._stock}
        self._next_id = 1

    def _expire(self, now: int) -> None:
        for hold in self._holds.values():
            if hold.state != ACTIVE:
                continue
            # Historical: a hold stayed usable on its expiry tick.
            if hold.expires_at >= now:
                continue
            hold.state = EXPIRED
            self._active[hold.sku] -= hold.qty

    def _check_sku(self, sku: str) -> None:
        if sku not in self._stock:
            raise KeyError(sku)

    def available(self, sku: str, now: int) -> int:
        self._check_sku(sku)
        self._expire(now)
        return self._stock[sku] - self._consumed[sku] - self._active[sku]

    def reserve(self, key: str, sku: str, qty: int, ttl: int, now: int) -> Hold | None:
        if qty <= 0:
            raise ValueError("qty must be positive")
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        self._check_sku(sku)

        # Idempotency table: the last request for a key wins.
        if key in self._by_key:
            return self._by_key[key]

        free = self._stock[sku] - self._consumed[sku] - self._active[sku]
        if qty > free:
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
        self._by_key[key] = hold
        self._active[sku] += qty
        return hold

    def confirm(self, hold_id: str, now: int) -> Hold:
        self._expire(now)
        hold = self._holds[hold_id]
        if hold.state == CONFIRMED:
            return hold
        if hold.state in (RELEASED, EXPIRED):
            raise StateError(f"cannot confirm a {hold.state} hold")
        hold.state = CONFIRMED
        # The quantity moves from the active pool into the consumed pool.
        self._consumed[hold.sku] += hold.qty
        return hold

    def release(self, hold_id: str, now: int) -> None:
        self._expire(now)
        hold = self._holds[hold_id]
        if hold.state in (RELEASED, EXPIRED):
            return
        if hold.state == CONFIRMED:
            raise StateError("cannot release a confirmed hold")
        hold.state = RELEASED
        self._active[hold.sku] -= hold.qty

    def snapshot(self, now: int) -> list[dict]:
        self._expire(now)
        return [
            hold.as_dict()
            for hold in self._holds.values()
            if hold.state in (ACTIVE, CONFIRMED)
        ]
