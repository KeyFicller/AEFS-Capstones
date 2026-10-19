"""Tool factories bound to a worktree."""

from pathlib import Path

from langchain_core.tools import BaseTool

from terminal_coding_agent.sandbox import DockerSandbox
from terminal_coding_agent.tools.clock import build_current_time
from terminal_coding_agent.tools.debug import build_debug
from terminal_coding_agent.tools.fs import build_edit_file, build_read_file
from terminal_coding_agent.tools.git import build_git
from terminal_coding_agent.tools.search import build_ripgrep, build_tree_sitter_symbols
from terminal_coding_agent.tools.shell import build_run_shell
from terminal_coding_agent.tools.web_search import build_web_search


def make_tools(
    worktree: Path,
    *,
    enable_web_search: bool = False,
    enable_debug: bool = False,
    shell_anything: bool = False,
    sandbox: DockerSandbox | None = None,
    allow_dangerous: bool = False,
) -> list[BaseTool]:
    """The worktree's tools. `web_search` and the lldb tools are opt-in.

    `shell_anything` replaces the set with `run_shell` alone. `sandbox` moves
    `run_shell` into a container; every other tool stays on the host.
    `allow_dangerous` lifts the git subcommand allowlist.
    """
    root = worktree.resolve()
    # With `--docker` the same files also answer to this path; accept either spelling.
    container_root = sandbox.container_root if sandbox is not None else None
    if shell_anything:
        return [build_run_shell(root, sandbox=sandbox)]
    tools = [
        build_read_file(root, container_root=container_root),
        build_edit_file(root, container_root=container_root),
        build_ripgrep(root, container_root=container_root),
        build_tree_sitter_symbols(root, container_root=container_root),
        build_run_shell(root, sandbox=sandbox),
        build_git(root, allow_dangerous=allow_dangerous),
        build_current_time(),
    ]
    if enable_web_search:
        tools.append(build_web_search())
    if enable_debug:
        tools.extend(build_debug(root))
    return tools


def tools_by_name(
    worktree: Path,
    *,
    enable_web_search: bool = False,
    enable_debug: bool = False,
    shell_anything: bool = False,
    sandbox: DockerSandbox | None = None,
    allow_dangerous: bool = False,
) -> dict[str, BaseTool]:
    return {
        tool.name: tool
        for tool in make_tools(
            worktree,
            enable_web_search=enable_web_search,
            enable_debug=enable_debug,
            shell_anything=shell_anything,
            sandbox=sandbox,
            allow_dangerous=allow_dangerous,
        )
    }
