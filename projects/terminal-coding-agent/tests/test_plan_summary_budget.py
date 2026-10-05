from langchain_core.messages import AIMessage
from terminal_coding_agent.ledger import BudgetSession


def test_budget_session_applies_usage_on_exit() -> None:
    msg = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 3,
            "total_tokens": 13,
            "input_token_details": {"cache_read": 2},
        },
    )
    with BudgetSession({}) as budget:
        budget.observe(msg)
    updates = budget.updates()
    assert updates["turns"] == 1
    assert updates["input_tokens"] == 10
    assert updates["output_tokens"] == 3
    assert updates["cache_read_tokens"] == 2
    assert updates["cost_rmb"] > 0
    assert updates["stop_reason"] is None


def test_budget_session_applies_usage_on_error() -> None:
    """A message the provider already billed counts even if the block later raised."""
    msg = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 3,
            "total_tokens": 13,
        },
    )
    try:
        with BudgetSession({}) as budget:
            budget.observe(msg)
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert budget.updates()["turns"] == 1
    assert budget.updates()["tokens"] == 13


def test_budget_session_sums_every_observed_message() -> None:
    """Summary makes more than one call; each must be billed, not just the last."""
    first = AIMessage(
        content="", usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
    )
    second = AIMessage(
        content="ok", usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}
    )

    with BudgetSession({}) as budget:
        budget.observe(first)
        budget.observe(second)

    updates = budget.updates()
    assert updates["turns"] == 2
    assert updates["tokens"] == 19
    assert updates["input_tokens"] == 15
    assert updates["output_tokens"] == 4
