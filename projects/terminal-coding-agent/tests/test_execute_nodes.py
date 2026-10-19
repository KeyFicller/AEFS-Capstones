"""Single execute node: middleware composition, blocked handling, budget hand-off."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware import ModelCallLimitMiddleware, TodoListMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from terminal_coding_agent import executor as executor_module
from terminal_coding_agent.executor import (
    _executor_prompt,
    _final_agent_message,
    build_execute_nodes,
)
from terminal_coding_agent.middleware import SequenceMiddleware
from terminal_coding_agent.middleware.ask_user import AskUserMiddleware
from terminal_coding_agent.middleware.safety import SafetyMiddleware
from terminal_coding_agent.middleware.todos import TodoMirrorMiddleware
from terminal_coding_agent.state import ToDoItem, ToDoStatus


def _base_state(**extra):
    state = {
        "messages": [],
        "todo_list": [],
        "turns": 0,
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cost_rmb": 0.0,
        "stop_reason": None,
        "blocked_reason": None,
    }
    state.update(extra)
    return state


def _nodes(**kwargs):
    return build_execute_nodes(MagicMock(), tools=[], **kwargs)


def _fake_agent(monkeypatch, *, result=None, error=None, captured=None):
    class FakeAgent:
        def invoke(self, payload, config):
            if captured is not None:
                captured["payload"] = payload
            if error is not None:
                raise error
            return result or {"messages": []}

    def fake_create_agent(**kwargs):
        if captured is not None:
            captured["system_prompt"] = kwargs["system_prompt"]
            captured["middleware"] = kwargs["middleware"]
        return FakeAgent()

    monkeypatch.setattr(executor_module, "create_agent", fake_create_agent)


def test_shell_anything_prompt_names_only_run_shell() -> None:
    prompt = _executor_prompt(enable_debug=True, shell_anything=True)

    assert "run_shell" in prompt
    assert "read_file" in prompt
    assert "debug_start" not in prompt


def test_planning_prompt_asks_for_todos_without_forcing_them() -> None:
    """The model may still skip the list; this only names the workflow."""
    prompt = _executor_prompt(enable_debug=False, shell_anything=False)

    assert "write_todos" in prompt
    assert "does not need a list" in prompt


def test_planning_prompt_is_absent_where_write_todos_does_not_exist() -> None:
    """`executor_only` and `chat` drop `TodoListMiddleware`; naming the tool would mislead."""
    assert "write_todos" not in _executor_prompt(
        enable_debug=False, shell_anything=False, executor_only=True
    )
    assert "write_todos" not in _executor_prompt(
        enable_debug=False, shell_anything=False, planning=False
    )


def test_shell_anything_does_not_rule_out_the_todo_list() -> None:
    prompt = _executor_prompt(enable_debug=False, shell_anything=True)

    assert "the todo list is still yours to use" in prompt


def test_dangerous_prompt_states_the_sandbox_is_open() -> None:
    """Without this the model will not try installs or network calls on its own."""
    closed = _executor_prompt(enable_debug=False, shell_anything=False)
    opened = _executor_prompt(enable_debug=False, shell_anything=False, allow_dangerous=True)
    shell_only = _executor_prompt(enable_debug=False, shell_anything=True, allow_dangerous=True)

    assert "no command restrictions" not in closed
    assert "no command restrictions" in opened
    # Orthogonal to the tool set: shell-only runs keep the addendum.
    assert "no command restrictions" in shell_only
    assert "run_shell" in shell_only


def test_final_agent_message_keeps_last_plain_ai() -> None:
    messages = [
        HumanMessage(content="do it"),
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "x"}, "id": "1"}],
        ),
        ToolMessage(content="file", tool_call_id="1"),
        AIMessage(content="fixed the typo"),
    ]
    final = _final_agent_message(messages)
    assert final is not None
    assert final.content == "fixed the typo"


def test_final_agent_message_drops_a_truncated_tool_call() -> None:
    messages = [
        AIMessage(content="earlier"),
        AIMessage(
            content="Now the main game module:",
            invalid_tool_calls=[
                {
                    "type": "invalid_tool_call",
                    "id": "call_1",
                    "name": "run_shell",
                    "args": '{"command": "cat',
                    "error": "Unterminated string",
                }
            ],
        ),
    ]

    final = _final_agent_message(messages)

    assert final is not None
    assert final.content == "Now the main game module:"
    assert not final.invalid_tool_calls
    assert not final.tool_calls


def test_run_agent_gets_the_whole_conversation(tmp_path: Path, monkeypatch) -> None:
    """One agent owns the turn, so it sees the turn's messages rather than a bare step."""
    captured: dict = {}
    _fake_agent(monkeypatch, captured=captured)
    state = _base_state(messages=[HumanMessage(content="fix the failing test")])

    _nodes()["run_agent"](state, {})

    assert captured["payload"]["messages"][0].content == "fix the failing test"
    assert "report_blocked" in captured["system_prompt"]


def test_run_agent_turns_a_block_report_into_a_stop_reason(tmp_path: Path, monkeypatch) -> None:
    _fake_agent(
        monkeypatch,
        result={"messages": [AIMessage(content="blocked")], "blocked_reason": "pytest fails"},
    )

    out = _nodes()["run_agent"](_base_state(), {})

    assert out["stop_reason"] == "blocked"
    assert out["blocked_reason"] == "pytest fails"


def test_run_agent_converts_model_crash_into_stop_reason(tmp_path: Path, monkeypatch) -> None:
    _fake_agent(monkeypatch, error=RuntimeError("deepseek 400"))

    out = _nodes()["run_agent"](_base_state(), {})

    assert out["stop_reason"] == "executor_error:RuntimeError"


def test_run_agent_turns_the_native_call_limit_into_max_turns(tmp_path: Path, monkeypatch) -> None:
    _fake_agent(
        monkeypatch,
        error=ModelCallLimitExceededError(
            thread_count=1, run_count=5, thread_limit=None, run_limit=5
        ),
    )

    out = _nodes()["run_agent"](_base_state(), {})

    assert out["stop_reason"] == "max_turns"


def test_run_agent_lets_a_graph_interrupt_through(tmp_path: Path, monkeypatch) -> None:
    """The pause must reach the parent graph, not become an executor_error (design §4)."""
    from langgraph.errors import GraphBubbleUp

    _fake_agent(monkeypatch, error=GraphBubbleUp())

    with pytest.raises(GraphBubbleUp):
        _nodes()["run_agent"](_base_state(), {})


def test_run_agent_publishes_the_agents_own_todos(tmp_path: Path, monkeypatch) -> None:
    """`TodoListMiddleware` owns the plan; the node forwards it for the trace and panel."""
    _fake_agent(
        monkeypatch,
        result={
            "messages": [AIMessage(content="done")],
            "todo_list": [ToDoItem(status=ToDoStatus.DONE, description="fix typo")],
        },
    )

    out = _nodes()["run_agent"](_base_state(), {})

    assert out["todo_list"][0].description == "fix typo"


def test_the_native_planner_is_installed_by_default(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}
    _fake_agent(monkeypatch, captured=captured)

    _nodes()["run_agent"](_base_state(), {})

    middleware = captured["middleware"]
    assert any(isinstance(m, TodoListMiddleware) for m in middleware)
    assert any(isinstance(m, ModelCallLimitMiddleware) for m in middleware)


def test_executor_only_drops_the_native_planner(tmp_path: Path, monkeypatch) -> None:
    """`executor_only` is the no-plan arm: no todo tool and no mirror, but the cap stays."""
    captured: dict = {}
    _fake_agent(monkeypatch, captured=captured)

    _nodes(executor_only=True)["run_agent"](_base_state(), {})

    middleware = captured["middleware"]
    assert not any(isinstance(m, TodoListMiddleware) for m in middleware)
    assert not any(isinstance(m, TodoMirrorMiddleware) for m in middleware)
    assert any(isinstance(m, ModelCallLimitMiddleware) for m in middleware)


def test_the_todo_panel_renderer_is_only_installed_for_a_planning_run(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict = {}
    _fake_agent(monkeypatch, captured=captured)
    render = MagicMock()

    _nodes(todo_render=render)["run_agent"](_base_state(), {})

    mirrors = [m for m in captured["middleware"] if isinstance(m, TodoMirrorMiddleware)]
    assert len(mirrors) == 1


@pytest.mark.parametrize("with_log", [True, False])
def test_run_agent_installs_the_sequence_logger_only_when_recording(
    tmp_path: Path, monkeypatch, with_log: bool
) -> None:
    """No `sequence_events` means no diagram was asked for, so nothing must be recorded."""
    captured: dict = {}
    _fake_agent(monkeypatch, captured=captured)
    events: list = []
    kwargs = {"sequence_events": events} if with_log else {}

    state = _base_state(messages=[HumanMessage(content="fix the failing test")])
    _nodes(**kwargs)["run_agent"](state, {})

    installed = [m for m in captured["middleware"] if isinstance(m, SequenceMiddleware)]
    assert bool(installed) is with_log
    if with_log:
        assert installed[0].events is events
        assert events[0] == ("task", "fix the failing test")


@pytest.mark.parametrize("with_hitl", [True, False])
def test_run_agent_installs_ask_user_only_when_hitl_is_enabled(
    tmp_path: Path, monkeypatch, with_hitl: bool
) -> None:
    """Harbor parity: without the flag the middleware table is unchanged."""
    captured: dict = {}
    _fake_agent(monkeypatch, captured=captured)
    kwargs = {"enable_hitl": with_hitl} if with_hitl else {}

    _nodes(**kwargs)["run_agent"](_base_state(), {})

    installed = [m for m in captured["middleware"] if isinstance(m, AskUserMiddleware)]
    assert bool(installed) is with_hitl


@pytest.mark.parametrize("allow_dangerous", [True, False])
def test_safety_guard_is_dropped_only_when_dangerous(
    tmp_path: Path, monkeypatch, allow_dangerous: bool
) -> None:
    captured: list = []

    class FakeAgent:
        def invoke(self, payload, config):
            return {"messages": []}

    def fake_create_agent(**kwargs):
        captured.append(kwargs["middleware"])
        return FakeAgent()

    monkeypatch.setattr(executor_module, "create_agent", fake_create_agent)
    nodes = _nodes(allow_dangerous=allow_dangerous)

    nodes["run_agent"](_base_state(), {})
    nodes["chat"](_base_state(messages=[HumanMessage(content="hi")]), {})

    for middleware in captured:
        installed = [m for m in middleware if isinstance(m, SafetyMiddleware)]
        assert bool(installed) is not allow_dangerous


def test_chat_node_carries_the_final_reply(tmp_path: Path, monkeypatch) -> None:
    """Summary relays the last message, so the chat node must end on its reply."""
    _fake_agent(monkeypatch, result={"messages": [AIMessage(content="the essay")]})

    out = _nodes()["chat"](_base_state(), {})

    assert out["messages"][0].content == "the essay"


def test_chat_node_runs_on_the_conversation_history(tmp_path: Path, monkeypatch) -> None:
    """Multi-turn memory: a follow-up must see the earlier turns, not a bare step."""
    captured: dict = {}
    _fake_agent(monkeypatch, captured=captured)
    state = _base_state(messages=[HumanMessage(content="first"), AIMessage(content="reply")])

    _nodes()["chat"](state, {})

    assert [m.content for m in captured["payload"]["messages"]] == ["first", "reply"]


def test_chat_node_has_no_planner(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}
    _fake_agent(monkeypatch, captured=captured)

    _nodes()["chat"](_base_state(), {})

    assert not any(isinstance(m, TodoListMiddleware) for m in captured["middleware"])


def test_a_stop_reason_ends_the_node_without_calling_the_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        executor_module,
        "create_agent",
        lambda **kwargs: pytest.fail("the model must not be reached"),
    )

    assert _nodes()["run_agent"](_base_state(stop_reason="max_turns"), {}) == {}


def test_attached_images_ride_along_in_the_turn_message(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}
    _fake_agent(monkeypatch, captured=captured)
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    state = _base_state(messages=[HumanMessage(content=[{"type": "text", "text": "look"}, image])])

    _nodes()["run_agent"](state, {})

    assert captured["payload"]["messages"][0].content == [
        {"type": "text", "text": "look"},
        image,
    ]
