from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from terminal_coding_agent.middleware.sequence import (
    SequenceMiddleware,
    _call_preview,
    _task_label,
    sequence_diagram,
    write_sequence_png,
)


def test_sequence_diagram_shows_calls_returns_and_a_final_reply() -> None:
    diagram = sequence_diagram(
        [
            ("call", "run_shell"),
            ("return", "run_shell", "error"),
            ("call", "report_blocked"),
            ("reply",),
        ]
    )

    assert diagram.index("Model->>run_shell: call") < diagram.index("run_shell-->>Model: error")
    assert "Model->>report_blocked: call" in diagram
    assert "report_blocked-->>Model" not in diagram
    assert "Note over Model: reply" in diagram
    assert "participant run_shell" in diagram
    assert "participant report_blocked" in diagram


def test_sequence_middleware_skips_work_without_a_shared_log() -> None:
    middleware = SequenceMiddleware(None)
    message = AIMessage(
        content="",
        tool_calls=[{"name": "run_shell", "args": {}, "id": "1"}],
    )

    assert middleware.after_model({"messages": [message]}, None) is None
    assert middleware.events is None


def test_sequence_diagram_boxes_each_task() -> None:
    long = "Edit greeter.py so greet returns Hello and replace SystemExit with print"
    diagram = sequence_diagram(
        [
            ("task", "Run python3 greeter.py"),
            ("call", "run_shell"),
            ("return", "run_shell", "success"),
            ("task", long),
            ("call", "edit_file"),
            ("return", "edit_file", "success"),
            ("reply",),
        ]
    )

    first_end = diagram.index("\n    end\n")
    assert diagram.index("rect rgb") < diagram.index("Run python3 greeter.py") < first_end
    assert diagram.index("Model->>run_shell: call") < first_end
    assert diagram.index("run_shell-->>Model: success") < first_end
    assert diagram.index("Model->>edit_file: call") > first_end
    assert diagram.index("Note over Model: reply") > first_end
    assert long not in diagram
    assert _task_label(long).endswith("...")
    assert _task_label(long) in diagram
    assert diagram.count("rect rgb") == 2
    assert diagram.count("\n    end\n") == 2


def test_call_preview_truncates_long_args() -> None:
    command = "python3 greeter.py && echo " + "x" * 120
    preview = _call_preview({"command": command})
    diagram = sequence_diagram([("call", "run_shell", preview)])

    assert preview.endswith("...")
    assert f"Model->>run_shell: call({preview})" in diagram
    assert command not in diagram


def test_one_invoke_writes_one_png(tmp_path: Path) -> None:
    events: list[tuple[str, ...]] = []
    first = SequenceMiddleware(events, description="Run python3 greeter.py")
    second = SequenceMiddleware(events, description="Edit greeter.py")
    first.after_model(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "run_shell", "args": {"command": "pytest"}, "id": "1"}],
                )
            ]
        },
        None,
    )
    first.wrap_tool_call(
        SimpleNamespace(tool_call={"name": "run_shell", "id": "1", "args": {}}),
        lambda request: ToolMessage(content="nope", tool_call_id="1", status="error"),
    )
    second.after_model(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "edit_file", "args": {"path": "a.py"}, "id": "2"}],
                )
            ]
        },
        None,
    )

    written: list[str] = []

    def fake_png(syntax: str, **kwargs):
        written.append(syntax)
        return b"\x89PNG\r\n\x1a\nfake"

    png_path = tmp_path / "sequence.png"
    with patch("terminal_coding_agent.middleware.sequence.draw_mermaid_png", fake_png):
        write_sequence_png(events, png_path)

    assert png_path.read_bytes().startswith(b"\x89PNG")
    assert written[0].index("run_shell") < written[0].index("edit_file")
    assert "call(pytest)" in written[0]
    assert "call(a.py)" in written[0]
    assert not (tmp_path / "task-0.png").exists()
