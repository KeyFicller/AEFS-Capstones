import difflib
import os
import tempfile
from pathlib import Path

from langchain_core.tools import BaseTool, tool
from rich.console import Console
from rich.syntax import Syntax

from terminal_coding_agent.tools.path import resolve_in_worktree
from terminal_coding_agent.tools.truncate import truncate

# Debug-only: flip to True to pretty-print edit diffs to the console.
DEBUG_RENDER_DIFF = False


def _write_atomic(path: Path, content: str) -> None:
    """Replace `path` through a sibling temp file, keeping its permission bits.

    A killed process (budget, timeout, crash) must not leave a half-written source
    file behind, which a plain `write_text` truncates first and rewrites after.
    """
    mode = path.stat().st_mode
    handle, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(content)
        tmp_path.chmod(mode)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def build_read_file(worktree: Path) -> BaseTool:
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


def build_edit_file(worktree: Path) -> BaseTool:
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

        if not old_str:
            # `"".count("")` is 1, so an empty needle would slip past the uniqueness
            # check on an empty file and insert at the start.
            return truncate("Error: old_str must not be empty")

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
            _write_atomic(resolved, new_content)
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
