import subprocess
import sys
from pathlib import Path

from budget import Budget
from terminal_coding_agent.ledger import write_trace


def test_script_directory_does_not_shadow_the_budget_package() -> None:
    pkg = Path(__file__).resolve().parents[1] / "src" / "terminal_coding_agent"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from terminal_coding_agent.ledger import ledger_from_state\n"
            "import budget\n"
            "print(budget.__file__)\n",
            str(pkg),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "terminal_coding_agent/budget.py" not in proc.stdout


def test_write_trace(tmp_path: Path):
    path = tmp_path / ".agent" / "trace.json"
    led = Budget(turns=3, cost_rmb=0.01, stop_reason="completed")
    write_trace(
        path,
        led,
        todo_list=[{"description": "x", "status": "DONE"}],
        stop_reason="completed",
    )
    assert path.is_file()
    text = path.read_text()
    assert "completed" in text
    assert "DEEPSEEK" not in text.upper()


def test_summarize_todos():
    from terminal_coding_agent.ledger import summarize_todos
    from terminal_coding_agent.state import ToDoItem, ToDoStatus

    items = [
        ToDoItem(status=ToDoStatus.DONE, description="fix typo"),
        ToDoItem(status=ToDoStatus.PENDING, description="verify"),
    ]
    assert summarize_todos(items) == [
        {"description": "fix typo", "status": "DONE"},
        {"description": "verify", "status": "PENDING"},
    ]


def test_ledger_state_roundtrip():
    from terminal_coding_agent.ledger import budget_updates, ledger_from_state

    state = {
        "turns": 2,
        "tokens": 10,
        "input_tokens": 8,
        "output_tokens": 2,
        "cache_read_tokens": 3,
        "cost_rmb": 0.001,
        "stop_reason": None,
    }
    led = ledger_from_state(state)
    assert led.turns == 2
    assert budget_updates(led)["input_tokens"] == 8
