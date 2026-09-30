from terminal_coding_agent.state import ToDoItem, ToDoStatus, replace_todos


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
