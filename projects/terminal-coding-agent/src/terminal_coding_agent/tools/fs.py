import difflib
from pathlib import Path

from langchain_core.tools import tool
from rich.console import Console
from rich.syntax import Syntax

from terminal_coding_agent.tools.path import resolve_in_worktree
from terminal_coding_agent.tools.truncate import truncate

# Debug-only: flip to True to pretty-print edit diffs to the console.
DEBUG_RENDER_DIFF = False


def build_read_file(worktree: Path):
    @tool
    def read_file(path: str) -> str:
        """Read a UTF-8 text file under the worktree.

        Args:
            path: The path to the file to read, relative to the worktree.

        Returns:
            The contents of the file, truncated to ~4000 tokens.
        """
        resolved = resolve_in_worktree(worktree, path)
        if isinstance(resolved, str):
            return truncate(resolved)
        try:
            return truncate(resolved.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            return truncate(f"Error: {exc}")

    return read_file


def build_edit_file(worktree: Path):
    @tool
    def edit_file(path: str, old_str: str, new_str: str) -> str:
        """Replace a unique old_str with new_str; return unified diff.

        Args:
            path: The path to the file to edit, relative to the worktree.
            old_str: The string to replace.
            new_str: The string to replace it with.

        Returns:
            The unified diff of the edits, truncated to ~4000 tokens.
        """
        resolved = resolve_in_worktree(worktree, path)
        if isinstance(resolved, str):
            return truncate(resolved)
        if not resolved.exists():
            return truncate(f"Error: file not found: {path}")

        try:
            content = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return truncate(f"Error: {exc}")

        count = content.count(old_str)
        if count == 0:
            return truncate(f"Error: old_str not found in {path}")
        if count > 1:
            return truncate(f"Error: old_str matched {count} times in {path}; must be unique")

        new_content = content.replace(old_str, new_str, 1)
        try:
            resolved.write_text(new_content, encoding="utf-8")
        except OSError as exc:
            return truncate(f"Error: {exc}")

        diff = "\n".join(
            difflib.unified_diff(
                content.splitlines(),
                new_content.splitlines(),
                fromfile=path,
                tofile=path,
                lineterm="",
            )
        )
        if DEBUG_RENDER_DIFF:
            Console().print(Syntax(diff, "diff"))
        return truncate(diff)

    return edit_file
