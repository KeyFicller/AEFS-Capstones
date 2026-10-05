from types import SimpleNamespace

from budget import Budget, usage_from_message


def test_spend_call_does_not_touch_turns() -> None:
    budget = Budget()
    budget.spend_call()
    assert budget.calls == 1
    assert budget.turns == 0
    assert budget.tokens == 0


def test_apply_usage_does_not_touch_calls() -> None:
    budget = Budget(price_cache_miss_per_m=1.0, price_output_per_m=4.0)
    budget.apply_usage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=0)
    assert budget.calls == 0
    assert budget.turns == 1
    assert budget.tokens == 2_000_000
    assert budget.cost_rmb == 5.0


def test_none_caps_do_not_trip() -> None:
    budget = Budget(calls=10, turns=10, tokens=10, cost_rmb=10.0)
    assert budget.check() is None
    assert budget.exhausted() is False


def test_check_order() -> None:
    budget = Budget(max_calls=1, max_turns=1, calls=1, turns=1)
    assert budget.check() == "max_calls"
    budget.max_calls = None
    assert budget.check() == "max_turns"
    budget.max_turns = None
    budget.max_tokens = 1
    budget.tokens = 1
    assert budget.check() == "max_tokens"
    budget.max_tokens = None
    budget.max_cost_rmb = 1
    budget.cost_rmb = 1
    assert budget.check() == "max_cost"
    budget.max_cost_rmb = None
    budget.max_seconds = 0
    assert budget.check() == "max_seconds"


def test_callback_falls_back_when_total_tokens_is_missing() -> None:
    budget = Budget()
    result = SimpleNamespace(
        generations=[
            [
                SimpleNamespace(
                    message=SimpleNamespace(
                        usage_metadata={"input_tokens": 3, "output_tokens": 4}
                    )
                )
            ]
        ]
    )
    budget.callback().on_llm_end(result)
    assert budget.tokens == 7
    assert budget.calls == 0
    assert budget.turns == 0


def test_callback_skips_a_generation_without_usage() -> None:
    budget = Budget()
    result = SimpleNamespace(
        generations=[[SimpleNamespace(message=SimpleNamespace(usage_metadata=None))]]
    )
    budget.callback().on_llm_end(result)
    assert budget.tokens == 0


def test_cache_read_is_clamped_to_input() -> None:
    budget = Budget(price_cache_hit_per_m=2.0, price_cache_miss_per_m=10.0)
    budget.apply_usage(input_tokens=1_000_000, output_tokens=0, cache_read_tokens=5_000_000)
    assert budget.cache_read_tokens == 1_000_000
    assert budget.cost_rmb == 2.0


def test_apply_usage_without_tokens_still_counts_a_turn() -> None:
    budget = Budget()
    budget.apply_usage(input_tokens=0, output_tokens=0)
    assert budget.turns == 1
    assert budget.tokens == 0
    assert budget.cost_rmb == 0.0


def test_usage_from_message_reads_cache_read() -> None:
    message = SimpleNamespace(
        usage_metadata={
            "input_tokens": 8,
            "output_tokens": 2,
            "input_token_details": {"cache_read": 3},
        }
    )
    assert usage_from_message(message) == (8, 2, 3)
