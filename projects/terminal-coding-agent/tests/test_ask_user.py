"""ask_user: the model asks, and the harness pauses before any tool runs."""

from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.types import Command
from terminal_coding_agent.checkpoint import build_checkpointer
from terminal_coding_agent.middleware.ask_user import AskUserMiddleware
from terminal_coding_agent.state import CodingAgentState, ToDoItem, ToDoStatus

SIDE_EFFECTS: list[str] = []
BOUND_TOOLS: list[str] = []


@tool
def side_effect() -> str:
    """A mutating tool: must run exactly once across the pause/resume."""
    SIDE_EFFECTS.append("ran")
    return "side effect done"


class SpyModel(GenericFakeChatModel):
    """Records the tools it is offered, then answers without calling any."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        BOUND_TOOLS.clear()
        BOUND_TOOLS.extend(getattr(item, "name", None) for item in tools)
        return self


class ScriptedModel(BaseChatModel):
    """Asks on the first turn; once any ToolMessage exists, wraps up."""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        return self

    def _generate(  # noqa: ARG002
        self, messages, stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        if any(isinstance(message, ToolMessage) for message in messages):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ask_user",
                                "args": {"question": "which?", "options": ["a", "b"]},
                                "id": "c1",
                            },
                            {"name": "side_effect", "args": {}, "id": "c2"},
                        ],
                    )
                )
            ]
        )


def _app(tmp_path: Path, thread: str):
    agent = create_agent(
        model=ScriptedModel(),
        tools=[side_effect],
        middleware=[AskUserMiddleware()],
        checkpointer=build_checkpointer(tmp_path),
    )
    return agent, {"configurable": {"thread_id": thread}}


def test_ask_user_is_offered_to_the_model() -> None:
    """`middleware.tools` is how the model learns the tool exists."""
    model = SpyModel(messages=iter([AIMessage(content="hi")]))
    agent = create_agent(model=model, tools=[], middleware=[AskUserMiddleware()])

    agent.invoke({"messages": [HumanMessage(content="hello")]})

    assert BOUND_TOOLS == ["ask_user"]


def test_ask_user_pauses_the_graph(tmp_path: Path) -> None:
    SIDE_EFFECTS.clear()
    agent, config = _app(tmp_path, "pause")

    out = agent.invoke({"messages": [HumanMessage(content="go")]}, config)

    assert out["__interrupt__"][0].value == {
        "type": "question",
        "question": "which?",
        "options": ["a", "b"],
    }
    assert SIDE_EFFECTS == [], "nothing may run before the interrupt"


def test_answer_resumes_without_replaying_the_sibling(tmp_path: Path) -> None:
    """The sibling must run once, not twice: the interrupt lands before the tool node."""
    SIDE_EFFECTS.clear()
    agent, config = _app(tmp_path, "resume")
    agent.invoke({"messages": [HumanMessage(content="go")]}, config)

    out = agent.invoke(Command(resume={"answer": "b", "cancelled": False}), config)

    assert out["messages"][-1].content == "done"
    assert SIDE_EFFECTS == ["ran"]


def test_the_answer_reaches_the_model_as_a_tool_message(tmp_path: Path) -> None:
    agent, config = _app(tmp_path, "answer")
    agent.invoke({"messages": [HumanMessage(content="go")]}, config)

    out = agent.invoke(Command(resume={"answer": "b", "cancelled": False}), config)

    seen = [message.content for message in out["messages"] if isinstance(message, ToolMessage)]
    assert "User answered: b" in seen


def test_a_dismissed_question_lets_the_agent_carry_on(tmp_path: Path) -> None:
    """Ctrl-C is not a rejection: the model gets told to use its own judgement."""
    agent, config = _app(tmp_path, "dismissed")
    agent.invoke({"messages": [HumanMessage(content="go")]}, config)

    out = agent.invoke(Command(resume={"answer": None, "cancelled": True}), config)

    seen = [message.content for message in out["messages"] if isinstance(message, ToolMessage)]
    assert any("did not answer" in content for content in seen)
    assert out["messages"][-1].content == "done"


def test_no_ask_user_call_is_not_an_intervention() -> None:
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {"path": "x"}, "id": "1"}],
            )
        ]
    }

    assert AskUserMiddleware().after_model(state, None) is None


def test_too_many_options_error_without_pausing() -> None:
    """A malformed question is the model's mistake, not something to put to the user."""
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ask_user",
                        "args": {"question": "q", "options": ["a", "b", "c", "d", "e"]},
                        "id": "1",
                    }
                ],
            )
        ]
    }

    out = AskUserMiddleware().after_model(state, None)

    assert out is not None
    assert out["messages"][-1].status == "error"
    assert "1 to 4" in out["messages"][-1].content
    assert out["messages"][-1].tool_call_id == "1"
    assert state["messages"][0].tool_calls[0]["name"] == "ask_user", (
        "the call stays in place: the router needs it to loop back to the model"
    )


def test_only_the_first_question_of_a_batch_is_asked() -> None:
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "ask_user", "args": {"question": "q", "options": []}, "id": "1"},
                    {"name": "ask_user", "args": {"question": "q2", "options": ["b"]}, "id": "2"},
                ],
            )
        ]
    }

    out = AskUserMiddleware().after_model(state, None)

    contents = [message.content for message in out["messages"]]
    assert any("1 to 4" in content for content in contents)
    assert "Error: ask one question per step." in contents


def test_a_non_list_options_value_is_an_error_not_a_crash() -> None:
    """`options` is model output: iterating a number must not raise out of the middleware."""
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "ask_user", "args": {"question": "q", "options": 3}, "id": "1"}
                ],
            )
        ]
    }

    out = AskUserMiddleware().after_model(state, None)

    assert out is not None
    assert out["messages"][-1].status == "error"
    assert "1 to 4" in out["messages"][-1].content


def test_a_string_options_value_is_not_split_into_characters() -> None:
    """`options: "abc"` would become three options and pass the bound check unearned."""
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "ask_user", "args": {"question": "q", "options": "abc"}, "id": "1"}
                ],
            )
        ]
    }

    out = AskUserMiddleware().after_model(state, None)

    assert out is not None
    assert out["messages"][-1].status == "error"


def test_the_real_execute_node_surfaces_the_question_to_the_parent_graph(
    tmp_path: Path,
) -> None:
    """Composition check: the real node and the real middleware table, only the model is scripted."""
    from langgraph.graph import END, START, StateGraph
    from terminal_coding_agent.executor import build_execute_nodes
    from terminal_coding_agent.models import AgentModels

    SIDE_EFFECTS.clear()
    models = AgentModels(planner=ScriptedModel(), executor=ScriptedModel())
    nodes = build_execute_nodes(models, [side_effect], worktree=tmp_path, enable_hitl=True)

    parent = StateGraph(CodingAgentState)
    parent.add_node("run_agent", nodes["run_agent"])
    parent.add_edge(START, "run_agent")
    parent.add_edge("run_agent", END)
    app = parent.compile(checkpointer=build_checkpointer(tmp_path))
    config = {"configurable": {"thread_id": "execute-node"}}

    first = app.invoke(
        {
            "messages": [],
            "todo_list": [ToDoItem(ToDoStatus.IN_PROGRESS, "fix the typo")],
            "current_task_index": 0,
            "turns": 0,
            "tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cost_rmb": 0.0,
        },
        config,
    )

    assert first["__interrupt__"][0].value["question"] == "which?"
    assert app.get_state(config).next == ("run_agent",)
    assert SIDE_EFFECTS == []

    app.invoke(Command(resume={"answer": "a", "cancelled": False}), config)

    assert SIDE_EFFECTS == ["ran"]


class SoloModel(BaseChatModel):
    """Only ever calls ask_user; wraps up once it has seen any ToolMessage."""

    @property
    def _llm_type(self) -> str:
        return "solo"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        return self

    def _generate(  # noqa: ARG002
        self, messages, stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        if any(isinstance(message, ToolMessage) for message in messages):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="used the answer"))]
            )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ask_user",
                                "args": {"question": "which?", "options": ["a", "b"]},
                                "id": "c1",
                            }
                        ],
                    )
                )
            ]
        )


def test_a_lone_question_still_loops_back_to_the_model(tmp_path: Path) -> None:
    """The common case: no sibling tool, so the model must still get to use the answer.

    The answer arrives as a ToolMessage for a call left in `tool_calls`, which is how the
    agent's router learns to loop back to the model instead of ending on "no tool calls".
    """
    agent = create_agent(
        model=SoloModel(),
        tools=[],
        middleware=[AskUserMiddleware()],
        checkpointer=build_checkpointer(tmp_path),
    )
    config = {"configurable": {"thread_id": "solo"}}
    agent.invoke({"messages": [HumanMessage(content="go")]}, config)

    out = agent.invoke(Command(resume={"answer": "a", "cancelled": False}), config)

    assert out["messages"][-1].content == "used the answer"
