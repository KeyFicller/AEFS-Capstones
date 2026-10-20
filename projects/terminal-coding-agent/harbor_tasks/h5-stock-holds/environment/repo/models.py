"""Types shared by the reservation engine."""

from dataclasses import dataclass


class ConflictError(Exception):
    """An idempotency key was reused with different arguments."""


class StateError(Exception):
    """The hold is not in a state that allows this operation."""


@dataclass
class Hold:
    id: str
    key: str
    sku: str
    qty: int
    created_at: int
    expires_at: int
    state: str = "active"

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "key": self.key,
            "sku": self.sku,
            "qty": self.qty,
            "expires_at": self.expires_at,
            "state": self.state,
        }
