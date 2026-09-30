from terminal_coding_agent.graph import make_graph
from pathlib import Path
import os

def test_make_graph(tmp_path: Path) -> None:
    # Just for testing
    os.environ["DEEPSEEK_API_KEY"] = "test"
    agent = make_graph({
        "configurable": {
                "worktree": tmp_path,
            },
    })
    os.environ.pop("DEEPSEEK_API_KEY")

    assert agent is not None