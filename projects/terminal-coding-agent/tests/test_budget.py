from pathlib import Path

from terminal_coding_agent.budget import (
    BudgetLedger,
    apply_usage,
    check,
    price_usage,
    write_trace,
)
from terminal_coding_agent import config


def test_price_usage_includes_cache_read():
    # 1M cache hit + 0 miss + 0 out
    assert price_usage(input_tokens=1_000_000, output_tokens=0, cache_read_tokens=1_000_000) == config.PRICE_CACHE_HIT_PER_M
    # 1M miss + 1M out
    cost = price_usage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=0)
    assert cost == config.PRICE_CACHE_MISS_PER_M + config.PRICE_OUTPUT_PER_M


def test_check_trips_each_fuse():
    assert check(BudgetLedger(turns=config.MAX_TURNS)) == "max_turns"
    assert check(BudgetLedger(tokens=config.MAX_TOKENS)) == "max_tokens"
    assert check(BudgetLedger(cost_rmb=config.MAX_COST_RMB)) == "max_cost"
    assert check(BudgetLedger()) is None


def test_apply_usage_increments_turn_and_cost():
    led = BudgetLedger()
    apply_usage(led, input_tokens=1000, output_tokens=0, cache_read_tokens=0)
    assert led.turns == 1
    assert led.input_tokens == 1000
    assert led.cost_rmb > 0


def test_write_trace(tmp_path: Path):
    path = tmp_path / ".agent" / "trace.json"
    led = BudgetLedger(turns=3, cost_rmb=0.01, stop_reason="completed")
    write_trace(
        path,
        led,
        todo_list=[{"description": "x", "status": "DONE"}],
        stop_reason="completed",
    )
    assert path.is_file()
    text = path.read_text()
    assert "completed" in text
    assert "DEEPSEEK" not in text.upper()


def test_summarize_todos():
    from terminal_coding_agent.budget import summarize_todos
    from terminal_coding_agent.state import ToDoItem, ToDoStatus

    items = [
        ToDoItem(status=ToDoStatus.DONE, description="fix typo"),
        ToDoItem(status=ToDoStatus.PENDING, description="verify"),
    ]
    assert summarize_todos(items) == [
        {"description": "fix typo", "status": "DONE"},
        {"description": "verify", "status": "PENDING"},
    ]


def test_ledger_state_roundtrip():
    from terminal_coding_agent.budget import budget_updates, ledger_from_state

    state = {
        "turns": 2,
        "tokens": 10,
        "input_tokens": 8,
        "output_tokens": 2,
        "cache_read_tokens": 3,
        "cost_rmb": 0.001,
        "stop_reason": None,
    }
    led = ledger_from_state(state)
    assert led.turns == 2
    assert budget_updates(led)["input_tokens"] == 8