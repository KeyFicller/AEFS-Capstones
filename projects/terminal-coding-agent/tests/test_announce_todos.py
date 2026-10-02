from pathlib import Path
from unittest.mock import MagicMock

from terminal_coding_agent.executor import build_execute_nodes
from terminal_coding_agent.graph import _announce_todos
from terminal_coding_agent.state import ToDoItem, ToDoStatus

_TODOS = [
    ToDoItem(status=ToDoStatus.DONE, description="read"),
    ToDoItem(status=ToDoStatus.IN_PROGRESS, description="edit"),
]


def _state(**extra) -> dict:
    state = {"todo_list": [], "replan_count": 0}
    state.update(extra)
    return state


def test_announce_todos_prints_the_rewritten_list(capsys) -> None:
    node = _announce_todos(lambda state: {"todo_list": _TODOS})

    node(_state(), None)

    out = capsys.readouterr().out
    assert "[✓]  read" in out
    assert "[+]  edit" in out


def test_announce_todos_prints_nothing_without_a_rewrite(capsys) -> None:
    node = _announce_todos(lambda state: {"stop_reason": "max_turns"})

    node(_state(), None)

    assert capsys.readouterr().out == ""


def test_announce_todos_uses_the_injected_renderer(capsys) -> None:
    seen: list[tuple[str, int]] = []
    config = {"configurable": {"todo_renderer": lambda text, version: seen.append((text, version))}}

    _announce_todos(lambda state: {"todo_list": _TODOS})(_state(), config)

    assert any("[✓]  read" in text for text, _ in seen)
    assert capsys.readouterr().out == ""


def test_announce_todos_hands_the_replan_count_to_the_renderer() -> None:
    """The CLI puts this in the panel subtitle, so it must arrive separately from the text."""
    seen: list[int] = []
    config = {"configurable": {"todo_renderer": lambda text, version: seen.append(version)}}

    _announce_todos(lambda state: {"todo_list": _TODOS, "replan_count": 2})(
        _state(replan_count=0), config
    )

    assert seen == [2]


def test_announce_todos_prefers_the_replan_count_from_the_update(capsys) -> None:
    node = _announce_todos(lambda state: {"todo_list": _TODOS, "replan_count": 2})

    node(_state(replan_count=0), None)

    assert "replan v2" in capsys.readouterr().out


def test_announce_todos_forwards_config_to_a_two_argument_node() -> None:
    seen: list[object] = []
    marker = {"configurable": {"todo_renderer": lambda _text, _version: None}}

    def node(state, config):
        seen.append(config)
        return {"todo_list": _TODOS}

    _announce_todos(node)(_state(), marker)

    assert seen == [marker]


def test_announce_todos_returns_the_node_updates_unchanged() -> None:
    updates = {"todo_list": _TODOS, "stop_reason": None}
    node = _announce_todos(lambda state: updates)

    assert node(_state(), None) is updates


def _execute_state(todos: list[ToDoItem]) -> dict:
    return {
        "messages": [],
        "todo_list": todos,
        "turns": 0,
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cost_rmb": 0.0,
        "stop_reason": None,
        "current_task_index": None,
    }


def test_wrapping_the_real_execute_nodes_announces_each_status_change(
    tmp_path: Path, capsys
) -> None:
    """The wrapper must work against the real nodes, not just a lambda."""
    nodes = build_execute_nodes(MagicMock(), tools=[], worktree=tmp_path)
    state = _execute_state([ToDoItem(status=ToDoStatus.PENDING, description="fix typo")])

    started = _announce_todos(nodes["start_task"])(state, None)
    assert "[+]  fix typo" in capsys.readouterr().out

    _announce_todos(nodes["end_task"])({**state, **started}, None)
    assert "[✓]  fix typo" in capsys.readouterr().out
