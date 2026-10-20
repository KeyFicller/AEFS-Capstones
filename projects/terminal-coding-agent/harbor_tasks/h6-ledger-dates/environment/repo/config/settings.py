"""Billing settings.

The rounding mode is deployment configuration: the settlement export overrides
it through `LEDGER_ROUNDING_MODE`, while the API path relies on the default.
"""

import os
from dataclasses import dataclass

PER_INVOICE = "per_invoice"
PER_LINE = "per_line"
ROUNDING_MODES = (PER_INVOICE, PER_LINE)

# 2023 export job default; it still sets LEDGER_ROUNDING_MODE explicitly.
_LEGACY_DEFAULT = PER_LINE


@dataclass(frozen=True)
class Settings:
    rounding_mode: str = PER_INVOICE

    def __post_init__(self) -> None:
        if self.rounding_mode not in ROUNDING_MODES:
            raise ValueError(f"unknown rounding mode: {self.rounding_mode!r}")


def from_env(environ: dict | None = None) -> Settings:
    """Build settings from the environment, falling back to the legacy default."""
    env = os.environ if environ is None else environ
    return Settings(rounding_mode=env.get("LEDGER_ROUNDING_MODE", _LEGACY_DEFAULT))
