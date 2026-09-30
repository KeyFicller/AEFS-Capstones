
from __future__ import annotations

from pathlib import Path

def resolve_in_worktree(worktree: Path, user_path: str) -> Path | str:
    if not user_path or not user_path.strip():
        return "Error: path is empty"

    root = worktree.resolve()
    raw = Path(user_path)

    target = raw.resolve() if raw.is_absolute() else (root / raw).resolve()

    try:
        target.relative_to(root)
    except ValueError:
        return f"Error: path escapes worktree: {user_path}"

    return target