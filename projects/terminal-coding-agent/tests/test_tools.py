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


def test_shell_anything_is_only_run_shell(tmp_path: Path) -> None:
    names = {
        t.name
        for t in make_tools(
            tmp_path, shell_anything=True, enable_web_search=True, enable_debug=True
        )
    }

    assert names == {"run_shell"}
    assert set(tools_by_name(tmp_path, shell_anything=True)) == {"run_shell"}


class _FakeSandbox:
    container_root = "/workspace"

    def run(self, command: str, timeout: int):  # noqa: ARG002
        return 0, "from-container", ""


def test_make_tools_hands_the_sandbox_to_run_shell(tmp_path: Path) -> None:
    tools = tools_by_name(tmp_path, sandbox=_FakeSandbox())

    out = tools["run_shell"].invoke({"command": "true"})

    assert "from-container" in out


def test_file_tools_accept_the_container_path(tmp_path: Path) -> None:
    """A model that copies `/workspace/x` out of a shell result must still reach the file."""
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "a.py").write_text("def f():\n    pass\n", encoding="utf-8")
    tools = tools_by_name(tmp_path, sandbox=_FakeSandbox())

    assert "hello" in tools["read_file"].invoke({"path": "/workspace/a.txt"})
    assert "hello" in tools["ripgrep"].invoke({"pattern": "hello", "path": "/workspace/a.txt"})
    assert "function f" in tools["tree_sitter_symbols"].invoke({"path": "/workspace/a.py"})

    diff = tools["edit_file"].invoke(
        {"path": "/workspace/a.txt", "old_str": "hello", "new_str": "bye"}
    )
    assert "-hello" in diff and "+bye" in diff
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "bye"


def test_make_tools_passes_allow_dangerous_to_git(tmp_path: Path) -> None:
    out = tools_by_name(tmp_path, allow_dangerous=True)["git"].invoke(
        {"git_args": ["reset", "--hard"]}
    )

    assert "not allowed" not in out.lower()


def test_shell_anything_keeps_the_sandbox(tmp_path: Path) -> None:
    tools = tools_by_name(tmp_path, shell_anything=True, sandbox=_FakeSandbox())

    assert "from-container" in tools["run_shell"].invoke({"command": "true"})


def test_debug_does_not_disturb_web_search(tmp_path: Path) -> None:
    both = {t.name for t in make_tools(tmp_path, enable_web_search=True, enable_debug=True)}

    assert "web_search" in both
    assert "debug_start" in both
    assert "web_search" not in tools_by_name(tmp_path)
