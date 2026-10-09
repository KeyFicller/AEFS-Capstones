"""Tool factories bound to a worktree."""

from pathlib import Path

from langchain_core.tools import BaseTool

from terminal_coding_agent.tools.clock import build_current_time
from terminal_coding_agent.tools.debug import build_debug
from terminal_coding_agent.tools.fs import build_edit_file, build_read_file
from terminal_coding_agent.tools.git import build_git
from terminal_coding_agent.tools.search import build_ripgrep, build_tree_sitter_symbols
from terminal_coding_agent.tools.shell import build_run_shell
from terminal_coding_agent.tools.web_search import build_web_search


def make_tools(
    worktree: Path, *, enable_web_search: bool = False, enable_debug: bool = False
) -> list[BaseTool]:
    """The worktree's tools. `web_search` and the lldb tools are opt-in."""
    root = worktree.resolve()
    tools = [
        build_read_file(root),
        build_edit_file(root),
        build_ripgrep(root),
        build_tree_sitter_symbols(root),
        build_run_shell(root),
        build_git(root),
        build_current_time(),
    ]
    if enable_web_search:
        tools.append(build_web_search())
    if enable_debug:
        tools.extend(build_debug(root))
    return tools


def tools_by_name(
    worktree: Path, *, enable_web_search: bool = False, enable_debug: bool = False
) -> dict[str, BaseTool]:
    return {
        tool.name: tool
        for tool in make_tools(
            worktree, enable_web_search=enable_web_search, enable_debug=enable_debug
        )
    }
