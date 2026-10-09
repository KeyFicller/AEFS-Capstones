import ast
import subprocess
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from terminal_coding_agent.config import SHELL_TIMEOUT_SECONDS
from terminal_coding_agent.tools.display import tool_output
from terminal_coding_agent.tools.path import resolve_in_worktree
from terminal_coding_agent.tools.truncate import truncate


def build_ripgrep(worktree: Path) -> BaseTool:
    def _display(params: dict, result: str) -> tuple[str, str]:
        count = len(result.splitlines())
        unit = "hit" if count == 1 else "hits"
        return f"{count} {unit}", f"```text\n{result}\n```"

    @tool
    @tool_output(_display)
    def ripgrep(pattern: str, path: str = ".") -> str:
        """Search for a pattern in a file or directory using ripgrep.

        Args:
            pattern: The pattern to search for.
            path: The path to the file or directory to search, relative to the worktree.

        Returns:
            The search results, truncated to ~4000 tokens.
        """
        resolved = resolve_in_worktree(worktree, path)
        if isinstance(resolved, str):
            return truncate(resolved)
        if not resolved.exists():
            return truncate(f"Error: path not found: {path}")

        try:
            results = subprocess.run(
                [
                    "rg",
                    "--line-number",
                    "--color",
                    "never",
                    "--",
                    pattern,
                    str(resolved),
                ],
                cwd=worktree,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=SHELL_TIMEOUT_SECONDS,
            )
        except FileNotFoundError:
            return truncate("Error: rg not found")
        except subprocess.TimeoutExpired:
            return truncate(f"Error: rg timed out after {SHELL_TIMEOUT_SECONDS}s")
        except OSError as exc:
            return truncate(f"Error: could not run rg: {exc}")

        if results.returncode in (0, 1):
            return truncate(results.stdout)
        return truncate(f"Error: rg failed (exit {results.returncode}): {results.stderr}")

    return ripgrep


def build_tree_sitter_symbols(worktree: Path) -> BaseTool:
    def _display(params: dict, result: str) -> tuple[str, str]:
        count = len(result.splitlines())
        unit = "symbol" if count == 1 else "symbols"
        return f"{count} {unit}", f"```text\n{result}\n```"

    @tool
    @tool_output(_display)
    def tree_sitter_symbols(path: str) -> str:
        """List functions/classes in a Python file (MVP: stdlib ast backend).

        Args:
            path: The path to the file, relative to the worktree.

        Returns:
            Symbol listing, truncated to ~4000 tokens.
        """
        resolved = resolve_in_worktree(worktree, path)
        if isinstance(resolved, str):
            return truncate(resolved)
        if not resolved.exists():
            return truncate(f"Error: file not found: {path}")
        if resolved.suffix.lower() != ".py":
            return truncate(f"Error: unsupported language (MVP: Python only via ast): {path}")

        try:
            tree = ast.parse(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as exc:
            return truncate(f"Error: {exc}")

        symbols: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                symbols.append(f"function {node.name}:L{node.lineno}")
            elif isinstance(node, ast.AsyncFunctionDef):
                symbols.append(f"async function {node.name}:L{node.lineno}")
            elif isinstance(node, ast.ClassDef):
                symbols.append(f"class {node.name}:L{node.lineno}")
        return truncate("\n".join(symbols))

    return tree_sitter_symbols
