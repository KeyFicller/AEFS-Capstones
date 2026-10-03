"""Tool factories bound to a worktree."""

from pathlib import Path

from langchain_core.tools import BaseTool

from terminal_coding_agent.tools.fs import build_edit_file, build_read_file
from terminal_coding_agent.tools.git import build_git
from terminal_coding_agent.tools.search import build_ripgrep, build_tree_sitter_symbols
from terminal_coding_agent.tools.shell import build_run_shell


def make_tools(worktree: Path) -> list[BaseTool]:
    root = worktree.resolve()
    return [
        build_read_file(root),
        build_edit_file(root),
        build_ripgrep(root),
        build_tree_sitter_symbols(root),
        build_run_shell(root),
        build_git(root),
    ]


def tools_by_name(worktree: Path) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in make_tools(worktree)}
