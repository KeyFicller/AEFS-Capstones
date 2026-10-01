"""Tool-log middleware and its execute-node wiring."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

import terminal_coding_agent.executor as executor_module
from terminal_coding_agent.executor import build_execute_nodes
from terminal_coding_agent.middleware import ToolLogMiddleware
from terminal_coding_agent.state import ToDoItem, ToDoStatus


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return f"contents of {path}\nsecond line"


class _FakeModel(GenericFakeChatModel):
    """create_agent calls bind_tools; GenericFakeChatModel refuses, so accept and ignore."""

    def bind_tools(self, tools, **kwargs):
        return self


def _request(name: str = "read_file", args: dict | None = None):
    return SimpleNamespace(
        tool_call={"name": name, "args": args or {"path": "a.py"}, "id": "call-1"}
    )


def test_sync_forwards_name_args_and_result() -> None:
    seen: list[tuple] = []
    middleware = ToolLogMiddleware(lambda name, args, result: seen.append((name, args, result)))
    message = ToolMessage(content="file body", tool_call_id="call-1")

    result = middleware.wrap_tool_call(_request(), lambda request: message)

    assert result is message  # the renderer must not swallow the tool result
    assert seen == [("read_file", {"path": "a.py"}, "file body")]


def test_async_forwards_too() -> None:
    seen: list[tuple] = []
    middleware = ToolLogMiddleware(lambda name, args, result: seen.append((name, result)))

    async def handler(request):
        return ToolMessage(content="async body", tool_call_id="call-1")

    asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert seen == [("read_file", "async body")]


def test_a_missing_tool_call_does_not_crash() -> None:
    seen: list[str] = []
    middleware = ToolLogMiddleware(lambda name, args, result: seen.append(name))

    middleware.wrap_tool_call(SimpleNamespace(tool_call=None), lambda request: "raw")

    assert seen == ["unknown"]


def test_a_real_agent_run_drives_the_renderer() -> None:
    """Locks the contract with create_agent: name, args and result all arrive intact."""
    script = [
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "call-1"}],
        ),
        AIMessage(content="done"),
    ]
    seen: list[tuple] = []
    agent = create_agent(
        model=_FakeModel(messages=iter(script)),
        tools=[read_file],
        middleware=[ToolLogMiddleware(lambda name, args, result: seen.append((name, args, result)))],
    )

    agent.invoke({"messages": [{"role": "user", "content": "read a.py"}]})

    assert seen == [("read_file", {"path": "a.py"}, "contents of a.py\nsecond line")]


def _execute_state() -> dict:
    return {
        "messages": [],
        "todo_list": [ToDoItem(status=ToDoStatus.IN_PROGRESS, description="fix typo")],
        "turns": 0,
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cost_rmb": 0.0,
        "stop_reason": None,
        "current_task_index": 0,
    }


@pytest.mark.parametrize("with_renderer", [True, False])
def test_run_agent_installs_the_tool_log_only_with_a_renderer(
    tmp_path: Path, monkeypatch, with_renderer: bool
) -> None:
    """Harbor injects nothing, so its path must build exactly the same middleware."""
    captured: dict = {}

    class FakeAgent:
        def invoke(self, payload, config):
            return {"messages": []}

    def fake_create_agent(**kwargs):
        captured["middleware"] = kwargs["middleware"]
        return FakeAgent()

    monkeypatch.setattr(executor_module, "create_agent", fake_create_agent)

    kwargs = {"tool_renderer": lambda *_: None} if with_renderer else {}
    nodes = build_execute_nodes(MagicMock(), tools=[], worktree=tmp_path, **kwargs)
    nodes["run_agent"](_execute_state(), {"configurable": {}})

    installed = [m for m in captured["middleware"] if isinstance(m, ToolLogMiddleware)]
    assert bool(installed) is with_renderer
    if with_renderer:
        assert captured["middleware"][-1] is installed[0], "must wrap innermost"
