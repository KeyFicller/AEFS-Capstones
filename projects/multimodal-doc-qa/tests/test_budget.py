"""The budget is the only thing standing between a loop and an open-ended bill.

``max_rounds`` bounds *loops*, not spend: one round is up to three model calls, so a graph
that respects its loop bound can still cost more than a caller expected. These tests pin the
three caps independently, because a cap that only works when another one is off is not a cap.
"""

import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from multimodal_doc_qa.config import Settings
from multimodal_doc_qa.limits import Budget, BudgetCallback


def test_script_directory_does_not_shadow_the_budget_package() -> None:
    pkg = Path(__file__).resolve().parents[1] / "src" / "multimodal_doc_qa"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from multimodal_doc_qa.limits import Budget\n"
            "import budget\n"
            "print(budget.__file__)\n",
            str(pkg),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "multimodal_doc_qa/budget.py" not in proc.stdout


def _budget(**overrides: object) -> Budget:
    """A budget with every cap off unless the test turns one on."""
    caps: dict = {"max_calls": None, "max_tokens": None, "max_seconds": None}
    return Budget(**{**caps, **overrides})  # type: ignore[arg-type]


def _llm_result(*totals: int) -> object:
    """An ``LLMResult``-shaped object reporting ``total_tokens`` per generation."""
    return SimpleNamespace(
        generations=[
            [SimpleNamespace(message=SimpleNamespace(usage_metadata={"total_tokens": n}))]
            for n in totals
        ]
    )


# ------------------------------------------------------------------ one cap at a time


def test_no_caps_means_never_exhausted() -> None:
    budget = _budget()
    for _ in range(100):
        budget.spend_call()

    assert not budget.exhausted()


def test_the_call_cap_allows_exactly_that_many_calls() -> None:
    """``max_calls`` is a ceiling on calls *made*, so it must not refuse the Nth."""
    budget = _budget(max_calls=3)

    spent = []
    while not budget.exhausted():
        budget.spend_call()
        spent.append(budget.calls)

    assert spent == [1, 2, 3]


def test_the_token_cap_fires_at_the_cap_and_not_before() -> None:
    budget = _budget(max_tokens=100)

    BudgetCallback(budget).on_llm_end(_llm_result(60))
    assert not budget.exhausted()

    BudgetCallback(budget).on_llm_end(_llm_result(40))
    assert budget.exhausted()


def test_the_seconds_cap_fires_when_the_clock_catches_up() -> None:
    budget = _budget(max_seconds=0.05)

    assert not budget.exhausted()
    time.sleep(0.06)
    assert budget.exhausted()


def test_the_clock_starts_when_the_budget_is_made() -> None:
    """A cap that starts on the first call would let a slow model load go unbilled."""
    budget = _budget(max_seconds=0.2)

    time.sleep(0.25)

    assert budget.elapsed() >= 0.25
    assert budget.exhausted()


# ------------------------------------------------------------------ caps together


def test_a_spent_cap_exhausts_the_budget_even_when_others_are_fresh() -> None:
    """One spent cap is enough: an ask has to fit under all three, not under any one."""
    budget = _budget(max_calls=10, max_tokens=5)

    BudgetCallback(budget).on_llm_end(_llm_result(5))

    assert budget.calls == 0
    assert budget.exhausted()


# ------------------------------------------------------------------ the callback


def test_the_callback_sums_usage_across_batches_and_generations() -> None:
    budget = _budget()

    BudgetCallback(budget).on_llm_end(_llm_result(10, 20))
    BudgetCallback(budget).on_llm_end(_llm_result(5))

    assert budget.tokens == 35


def test_the_callback_ignores_generations_without_a_usage_report() -> None:
    """An unreported call must under-count, not invent a number or raise."""
    budget = _budget()
    result = SimpleNamespace(
        generations=[[SimpleNamespace(message=SimpleNamespace(usage_metadata=None))]]
    )

    BudgetCallback(budget).on_llm_end(result)

    assert budget.tokens == 0


def test_the_callback_falls_back_to_the_input_output_split() -> None:
    """Some providers never send ``total_tokens``; defaulting it to 0 would defeat the cap."""
    budget = _budget(max_tokens=100)
    usage = {"input_tokens": 60, "output_tokens": 40}
    result = SimpleNamespace(
        generations=[[SimpleNamespace(message=SimpleNamespace(usage_metadata=usage))]]
    )

    BudgetCallback(budget).on_llm_end(result)

    assert budget.tokens == 100
    assert budget.exhausted()


def test_from_settings_reads_all_three_caps() -> None:
    budget = Budget.from_settings(
        Settings(max_ask_calls=7, max_ask_tokens=1234, max_ask_seconds=9.0)
    )

    assert (budget.max_calls, budget.max_tokens, budget.max_seconds) == (7, 1234, 9.0)


def test_the_default_call_cap_covers_the_worst_case_ask() -> None:
    """The default has to admit the most expensive legal run, or it silently truncates it.

    Worst case is ``plan`` plus, per round, ``assess`` + ``synthesize`` + ``verify``:
    ``1 + 3 * max_rounds``.
    """
    settings = Settings()

    assert settings.max_ask_calls == 1 + settings.max_rounds * 3


def test_a_callback_shares_its_budget_with_the_caller() -> None:
    """The code counting calls and the code counting tokens must read one object."""
    budget = _budget(max_tokens=10)

    budget.callback().on_llm_end(_llm_result(10))

    assert budget.exhausted()
    assert budget.tokens == 10


def test_spend_call_counts_once_per_call() -> None:
    budget = _budget()

    budget.spend_call()
    budget.spend_call()

    assert budget.calls == pytest.approx(2)
