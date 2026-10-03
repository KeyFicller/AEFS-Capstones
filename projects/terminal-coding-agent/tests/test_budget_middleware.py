from pathlib import Path

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from terminal_coding_agent import config
from terminal_coding_agent.budget import BudgetLedger
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


def test_wrap_model_call_hard_stops_without_handler() -> None:
    ledger = BudgetLedger(turns=config.MAX_TURNS)
    middleware = BudgetMiddleware(ledger)
    called = {"n": 0}

    def handler(request):
        called["n"] += 1
        return AIMessage(content="should not run")

    out = middleware.wrap_model_call(request=_dummy_model_request(), handler=handler)

    assert called["n"] == 0
    assert isinstance(out, AIMessage)
    assert "budget exceeded" in out.content.lower()
    assert ledger.stop_reason == "max_turns"


def test_wrap_model_call_applies_usage_from_model_response() -> None:
    ledger = BudgetLedger()
    middleware = BudgetMiddleware(ledger)

    def handler(request):
        return _model_response_with_usage()

    result = middleware.wrap_model_call(request=_dummy_model_request(), handler=handler)

    assert isinstance(result, ModelResponse)
    assert ledger.turns == 1
    assert ledger.input_tokens == 100
    assert ledger.output_tokens == 20
    assert ledger.cache_read_tokens == 40
    assert ledger.cost_rmb > 0


def test_after_agent_does_not_write_trace(tmp_path: Path) -> None:
    trace_path = tmp_path / ".agent" / "trace.json"
    middleware = BudgetMiddleware(BudgetLedger(turns=1))
    assert middleware.after_agent(state={}, runtime=None) is None
    assert not trace_path.exists()
