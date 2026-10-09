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
        "current_time",
    }

    assert set(tools_by_name(tmp_path).keys()) == names


def test_current_time_is_always_available(tmp_path: Path) -> None:
    """Unlike `web_search`, the clock needs no flag: it is offline-only and dependency-free."""
    assert "current_time" in {t.name for t in make_tools(tmp_path)}
    assert "current_time" in {t.name for t in make_tools(tmp_path, enable_web_search=True)}


def test_web_search_is_opt_in(tmp_path: Path) -> None:
    default = {t.name for t in make_tools(tmp_path)}
    enabled = {t.name for t in make_tools(tmp_path, enable_web_search=True)}

    assert "web_search" not in default
    assert enabled == default | {"web_search"}


def test_tools_by_name_honours_the_flag(tmp_path: Path) -> None:
    assert "web_search" in tools_by_name(tmp_path, enable_web_search=True)
    assert "web_search" not in tools_by_name(tmp_path)


def test_debug_tools_are_opt_in(tmp_path: Path) -> None:
    default = {t.name for t in make_tools(tmp_path)}
    enabled = {t.name for t in make_tools(tmp_path, enable_debug=True)}
    debug_names = {"debug_start", "debug_cmd", "debug_stop"}

    assert not (debug_names & default)
    assert enabled == default | debug_names


def test_debug_does_not_disturb_web_search(tmp_path: Path) -> None:
    both = {t.name for t in make_tools(tmp_path, enable_web_search=True, enable_debug=True)}

    assert "web_search" in both
    assert "debug_start" in both
    assert "web_search" not in tools_by_name(tmp_path)
