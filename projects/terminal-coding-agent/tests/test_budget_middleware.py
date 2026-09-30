from __future__ import annotations

import json
from pathlib import Path

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

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
    ledger = BudgetLedger(turns=50)
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
    assert ledger.cost_usd > 0


def test_after_agent_writes_trace(tmp_path: Path) -> None:
    trace_path = tmp_path / ".agent" / "trace.json"
    ledger = BudgetLedger(
        turns=1,
        tokens=120,
        input_tokens=100,
        output_tokens=20,
        cache_read_tokens=40,
        cost_usd=0.0123,
    )
    middleware = BudgetMiddleware(
        ledger,
        trace_path=trace_path,
        todo_list=[{"description": "demo", "status": "DONE"}],
    )

    assert middleware.after_agent(state={}, runtime=None) is None

    assert trace_path.is_file()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["turns"] == 1
    assert payload["tokens"] == 120
    assert payload["input_tokens"] == 100
    assert payload["output_tokens"] == 20
    assert payload["cache_read_tokens"] == 40
    assert payload["cost_usd"] == 0.0123
    assert payload["stop_reason"] == "completed"
    assert payload["todo_list"] == [{"description": "demo", "status": "DONE"}]
    assert "finished_at" in payload
    assert "T" in payload["finished_at"]
