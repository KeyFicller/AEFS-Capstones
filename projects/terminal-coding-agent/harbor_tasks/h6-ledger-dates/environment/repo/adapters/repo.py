"""In-memory subscriptions, plans and credit balances."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Plan:
    code: str
    name: str
    monthly_fee_fen: int

    def __post_init__(self) -> None:
        if self.monthly_fee_fen < 0:
            raise ValueError("plan fee cannot be negative")


@dataclass(frozen=True)
class Addon:
    code: str
    monthly_fee_fen: int
    start_date: date
    end_date: date | None = None

    def __post_init__(self) -> None:
        if self.monthly_fee_fen < 0:
            raise ValueError("addon fee cannot be negative")
        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError("addon ends before it starts")


@dataclass(frozen=True)
class Subscription:
    tenant: str
    plan: Plan
    start_date: date
    end_date: date | None = None
    addons: tuple[Addon, ...] = ()

    def __post_init__(self) -> None:
        if not self.tenant:
            raise ValueError("tenant is required")
        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError("subscription ends before it starts")


class Repo:
    """Tiny stand-in for the billing database."""

    def __init__(self, subscriptions=(), credits: dict | None = None) -> None:
        self._subscriptions: dict[str, Subscription] = {}
        for subscription in subscriptions:
            self.add(subscription)
        self._credits = {tenant: int(fen) for tenant, fen in (credits or {}).items()}
        for tenant, fen in self._credits.items():
            if fen < 0:
                raise ValueError(f"negative credit for {tenant}")

    def add(self, subscription: Subscription) -> None:
        if subscription.tenant in self._subscriptions:
            raise ValueError(f"duplicate tenant: {subscription.tenant}")
        self._subscriptions[subscription.tenant] = subscription

    def get(self, tenant: str) -> Subscription:
        try:
            return self._subscriptions[tenant]
        except KeyError:
            raise KeyError(f"unknown tenant: {tenant}") from None

    def tenants(self) -> list[str]:
        return sorted(self._subscriptions)

    def credit_balance(self, tenant: str) -> int:
        return self._credits.get(tenant, 0)

    def set_credit_balance(self, tenant: str, fen: int) -> None:
        if fen < 0:
            raise ValueError("credit balance cannot be negative")
        self._credits[tenant] = fen
