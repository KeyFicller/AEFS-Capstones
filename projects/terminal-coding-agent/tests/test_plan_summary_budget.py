from __future__ import annotations

from langchain_core.messages import AIMessage

from terminal_coding_agent.budget import BudgetSession


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
    assert updates["cost_usd"] > 0
    assert updates["stop_reason"] is None


def test_budget_session_skips_usage_on_error() -> None:
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
    assert budget.updates()["turns"] == 0
