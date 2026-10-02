from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from terminal_coding_agent.middleware.recover import BlockedReportMiddleware


class _ToolFakeChatModel(GenericFakeChatModel):
    """Scripted model that tolerates bind_tools (create_agent calls it)."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ToolFakeChatModel:
        return self


def _state(*messages):
    return {"messages": list(messages)}


def test_report_blocked_is_registered_as_tool() -> None:
    assert [item.name for item in BlockedReportMiddleware().tools] == ["report_blocked"]


def test_after_model_jumps_to_end_and_records_reason() -> None:
    middleware = BlockedReportMiddleware()
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "report_blocked", "args": {"reason": "pytest fails on import"}, "id": "c1"},
            {"name": "run_shell", "args": {"command": "ls"}, "id": "c2"},
        ],
    )

    out = middleware.after_model(_state(message), None)

    assert out is not None
    assert out["jump_to"] == "end"
    assert out["blocked_reason"] == "pytest fails on import"
    tool_messages = {m.tool_call_id: m for m in out["messages"] if isinstance(m, ToolMessage)}
    assert set(tool_messages) == {"c1", "c2"}
    assert tool_messages["c1"].status == "success"
    assert "Blocked reported." in tool_messages["c1"].content
    assert tool_messages["c2"].status == "error"
    assert "Skipped" in tool_messages["c2"].content
    assert isinstance(out["messages"][-1], AIMessage)
    assert "pytest fails on import" in out["messages"][-1].content


def test_after_model_ignores_other_tool_calls() -> None:
    middleware = BlockedReportMiddleware()
    message = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "x.py"}, "id": "c1"}],
    )

    assert middleware.after_model(_state(message), None) is None


def test_after_model_with_no_messages_returns_none() -> None:
    assert BlockedReportMiddleware().after_model({"messages": []}, None) is None


def test_agent_stops_and_reports_blocked_without_running_tools() -> None:
    """Real create_agent run: jump_to="end" must skip the tool node entirely."""
    ran: list[str] = []

    @tool
    def probe(x: int) -> str:
        """A harmless side-effecting tool."""
        ran.append("probe")
        return "ran"

    scripted = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "report_blocked", "args": {"reason": "missing dependency"}, "id": "1"},
                    {"name": "probe", "args": {"x": 1}, "id": "2"},
                ],
            )
        ]
    )
    agent = create_agent(
        _ToolFakeChatModel(messages=scripted),
        tools=[probe],
        middleware=[BlockedReportMiddleware()],
    )

    out = agent.invoke({"messages": [HumanMessage(content="go")]})

    assert out["blocked_reason"] == "missing dependency"
    assert ran == []
    assert isinstance(out["messages"][-1], AIMessage)
    assert len([m for m in out["messages"] if isinstance(m, ToolMessage)]) == 2
