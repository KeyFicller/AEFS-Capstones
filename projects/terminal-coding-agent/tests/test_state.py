import terminal_coding_agent.state as state_mod
from terminal_coding_agent.state import ToDoItem, ToDoStatus, replace_todos, tag_replan_version


def test_replace_todos_detects_status_change_with_new_items() -> None:
    old = [ToDoItem(status=ToDoStatus.PENDING, description="fix typo")]
    new = [ToDoItem(status=ToDoStatus.DONE, description="fix typo")]
    assert replace_todos(old, new) is new


def test_replace_todos_snapshot_ignores_shared_object_trap() -> None:
    """Shallow-copy + in-place mutate used to make old == new; snapshot must still differ
    only when callers replace items. This documents the expected immutable update path."""
    shared = ToDoItem(status=ToDoStatus.PENDING, description="fix typo")
    old = [shared]
    new = [ToDoItem(status=ToDoStatus.IN_PROGRESS, description="fix typo")]
    assert replace_todos(old, new) is new
    assert shared.status == ToDoStatus.PENDING


def test_replace_todos_header_shows_replan_version(capsys) -> None:
    state_mod._last_printed = None
    new = tag_replan_version([ToDoItem(description="new")], 1)
    replace_todos([ToDoItem(description="old")], new)
    assert "replan v1" in capsys.readouterr().out


def test_replace_todos_prints_each_change_once(capsys) -> None:
    state_mod._last_printed = None
    old = [ToDoItem(description="a")]
    new = [ToDoItem(description="b")]
    replace_todos(old, new)
    replace_todos(old, new)
    assert capsys.readouterr().out.count("Execute State Update") == 1

    replace_todos(new, [ToDoItem(description="c")])
    assert capsys.readouterr().out.count("Execute State Update") == 1
