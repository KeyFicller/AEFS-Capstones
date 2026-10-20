"""HTTP-facing entry points (the transport layer lives outside this repo)."""

from adapters.repo import Repo, Subscription
from config.settings import Settings
from service.invoicing import build_invoice


def get_invoice(
    repo: Repo,
    tenant: str,
    period: str,
    settings: Settings | None = None,
) -> dict:
    """Return the invoice for `tenant` in `period`."""
    if not isinstance(tenant, str) or not tenant:
        raise TypeError("tenant must be a non-empty string")
    return build_invoice(repo, tenant, period, settings)


def list_tenants(repo: Repo) -> list[str]:
    return repo.tenants()


def get_subscription(repo: Repo, tenant: str) -> Subscription:
    return repo.get(tenant)
