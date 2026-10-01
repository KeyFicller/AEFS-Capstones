from terminal_coding_agent.state import ToDoItem, ToDoStatus, format_todos, replace_todos


def test_replace_todos_returns_the_new_list() -> None:
    old = [ToDoItem(status=ToDoStatus.PENDING, description="fix typo")]
    new = [ToDoItem(status=ToDoStatus.DONE, description="fix typo")]
    assert replace_todos(old, new) is new


def test_format_todos_shows_status_and_version() -> None:
    text = format_todos(
        [
            ToDoItem(status=ToDoStatus.DONE, description="read"),
            ToDoItem(status=ToDoStatus.FAILED, description="edit"),
        ],
        version=2,
    )
    assert "replan v2" in text
    assert "[✓]  read" in text
    assert "[✗]  edit" in text
