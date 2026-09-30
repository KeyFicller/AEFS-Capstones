"""Project-level constants and local environment loading."""

from __future__ import annotations

import os
from pathlib import Path

MODEL = "deepseek:deepseek-v4-flash"
SHELL_TIMEOUT_SECONDS = 60
TOOL_OUTPUT_TOKEN_LIMIT = 4000
CHARS_PER_TOKEN_APPROX = 4

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ENV_PATH = PROJECT_ROOT / "local.env"


def load_local_env(env_path: Path) -> None:
    """Load local.env into os.environ (existing keys are not overwritten)."""
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value
