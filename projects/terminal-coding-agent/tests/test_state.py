from terminal_coding_agent.state import ToDoItem, ToDoStatus, format_todos, replace_todos


def test_replace_todos_returns_the_new_list() -> None:
    old = [ToDoItem(status=ToDoStatus.PENDING, description="fix typo")]
    new = [ToDoItem(status=ToDoStatus.DONE, description="fix typo")]
    assert replace_todos(old, new) is new


def test_format_todos_shows_each_status() -> None:
    text = format_todos(
        [
            ToDoItem(status=ToDoStatus.DONE, description="read"),
            ToDoItem(status=ToDoStatus.FAILED, description="edit"),
        ]
    )
    assert "[✓]  read" in text
    assert "[✗]  edit" in text


def test_format_todos_leaves_the_box_and_replan_to_the_renderer() -> None:
    """ASCII chrome here would be drawn a second time once a real panel wraps it."""
    text = format_todos([ToDoItem(status=ToDoStatus.PENDING, description="read")])

    assert "#" not in text
    assert "replan" not in text
