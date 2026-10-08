from langchain_core.messages import AIMessage, HumanMessage
from terminal_coding_agent.gate import build_intent
from terminal_coding_agent.models import AgentModels


class _StubClassifierModel:
    """Stands in for the planner model the classifier binds."""

    def __init__(self, intent: str) -> None:
        self.intent = intent

    def with_structured_output(self, schema, include_raw=False):  # noqa: ARG002
        outer = self

        class _Structured:
            def invoke(self, messages):
                from intent import IntentVerdict

                raw = AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                )
                return {
                    "raw": raw,
                    "parsed": IntentVerdict(intent=outer.intent),
                    "parsing_error": None,
                }

        return _Structured()


def test_intent_node_writes_the_label_and_accounts_the_call() -> None:
    model = _StubClassifierModel("chat")
    node = build_intent(AgentModels(planner=model, executor=model))

    out = node(
        {"messages": [HumanMessage(content="hi")], "turns": 0, "tokens": 0, "stop_reason": None}
    )

    assert out["intent"] == "chat"
    assert out["turns"] == 1, "the intent call must reach the ledger"


def test_intent_node_defaults_to_work_when_classification_fails() -> None:
    class _Boom:
        def with_structured_output(self, schema, include_raw=False):  # noqa: ARG002
            class _Structured:
                def invoke(self, messages):
                    raise RuntimeError("provider down")

            return _Structured()

    model = _Boom()
    node = build_intent(AgentModels(planner=model, executor=model))

    out = node(
        {"messages": [HumanMessage(content="hi")], "turns": 0, "tokens": 0, "stop_reason": None}
    )

    assert out["intent"] == "work", "an unclear turn is work, never a crash"
