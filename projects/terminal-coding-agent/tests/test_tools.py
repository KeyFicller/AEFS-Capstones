from pathlib import Path

from terminal_coding_agent.tools import make_tools, tools_by_name


def test_make_tools(tmp_path: Path) -> None:
    names = {t.name for t in make_tools(tmp_path)}

    assert names == {
        "read_file",
        "edit_file",
        "ripgrep",
        "tree_sitter_symbols",
        "run_shell",
        "git",
    }

    assert set(tools_by_name(tmp_path).keys()) == names
