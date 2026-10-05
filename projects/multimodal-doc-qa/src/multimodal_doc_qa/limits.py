"""Per-ask spend caps. The ledger lives in the shared ``budget`` package."""

from budget import Budget as _Budget
from budget import BudgetCallback as BudgetCallback

from multimodal_doc_qa.config import Settings


class Budget(_Budget):
    """Doc-qa caps: calls, tokens, and wall clock. Other caps stay off."""

    @classmethod
    def from_settings(cls, settings: Settings) -> "Budget":
        return cls(
            max_calls=settings.max_ask_calls,
            max_tokens=settings.max_ask_tokens,
            max_seconds=settings.max_ask_seconds,
        )
