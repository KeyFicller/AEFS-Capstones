from pathlib import Path

from budget import Budget
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from terminal_coding_agent import config
from terminal_coding_agent.middleware.budget import BudgetMiddleware


def _dummy_model_request() -> dict:
    return {
        "model": "gpt-4o-mini",
        "messages": [HumanMessage(content="Hello, world!")],
        "config": {},
    }


def _model_response_with_usage(
    *,
    input_tokens: int = 100,
    output_tokens: int = 20,
    cache_read: int = 40,
) -> ModelResponse:
    return ModelResponse(
        result=[
            AIMessage(
                content="ok",
                usage_metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "input_token_details": {"cache_read": cache_read},
                },
            )
        ]
    )


def test_wrap_model_call_only_counts_spend() -> None:
    """Enforcement lives in `ModelCallLimitMiddleware`; this layer must not second-guess it."""
    ledger = Budget(max_turns=config.MAX_TURNS, turns=config.MAX_TURNS)
    middleware = BudgetMiddleware(ledger)
    called = {"n": 0}

    def handler(request):
        called["n"] += 1
        return AIMessage(
            content="ran",
            usage_metadata={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        )

    out = middleware.wrap_model_call(request=_dummy_model_request(), handler=handler)

    assert called["n"] == 1
    assert out.content == "ran"
    assert ledger.turns == config.MAX_TURNS + 1


def test_wrap_model_call_applies_usage_from_model_response() -> None:
    ledger = Budget()
    middleware = BudgetMiddleware(ledger)

    def handler(request):
        return _model_response_with_usage()

    result = middleware.wrap_model_call(request=_dummy_model_request(), handler=handler)

    assert isinstance(result, ModelResponse)
    assert ledger.turns == 1
    assert ledger.input_tokens == 100
    assert ledger.output_tokens == 20
    assert ledger.cache_read_tokens == 40
    assert ledger.cost_rmb == 0


def test_after_agent_does_not_write_trace(tmp_path: Path) -> None:
    trace_path = tmp_path / ".agent" / "trace.json"
    middleware = BudgetMiddleware(Budget(turns=1))
    assert middleware.after_agent(state={}, runtime=None) is None
    assert not trace_path.exists()
