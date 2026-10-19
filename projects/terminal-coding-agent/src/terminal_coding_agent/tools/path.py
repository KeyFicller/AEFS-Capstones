from pathlib import Path


def resolve_in_worktree(
    worktree: Path, user_path: str, *, container_root: str | None = None
) -> Path | str:
    if not user_path or not user_path.strip():
        return "Error: path is empty"

    root = worktree.resolve()
    raw = Path(user_path)

    # With `--docker` the same files are also visible at `container_root`; a model that
    # echoes a path printed by `pwd` or a traceback must not be told it escapes.
    if container_root and raw.is_absolute():
        mount = Path(container_root)
        if raw == mount or mount in raw.parents:
            raw = raw.relative_to(mount)

    target = raw.resolve() if raw.is_absolute() else (root / raw).resolve()

    try:
        target.relative_to(root)
    except ValueError:
        return f"Error: path escapes worktree: {user_path}"

    return target
