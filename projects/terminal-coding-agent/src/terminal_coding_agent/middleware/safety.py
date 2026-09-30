from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

# D3 denylist: structured patterns only (not naive substring keywords).
_DESTRUCTIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"(^|[\s;|&])rm\s+(-[^\s]*\s+)*/*(\s|$)",  # rm ... / or rm -rf /
        r"\brm\s+-[^\s]*r[^\s]*f[^\s]*\s+/",  # rm -rf / variants
        r"\brm\s+-[^\s]*f[^\s]*r[^\s]*\s+/",
        r"\bmkfs\b",
        r"\bdd\s+if=",
        r"\bcurl\b[^|\n]*\|\s*(sh|bash)\b",
        r"\bwget\b[^|\n]*\|\s*(sh|bash)\b",
        r":\(\)\s*\{\s*:\|:&\s*\}\s*;?",  # fork bomb
    )
)


def is_destructive_shell(command: str) -> bool:
    """Return True if the shell command matches the D3 denylist."""
    text = command.strip()
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    if re.search(r":\(\)\{:\|:&\};?", compact):
        return True
    return any(pattern.search(text) for pattern in _DESTRUCTIVE_PATTERNS)


class SafetyMiddleware(AgentMiddleware):
    """PreToolUse guard: block destructive run_shell commands (D3)."""

    def wrap_tool_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        name = request.tool_call.get("name")
        if name == "run_shell":
            command = str((request.tool_call.get("args") or {}).get("command", ""))
            if is_destructive_shell(command):
                return ToolMessage(
                    content=(
                        "Error: blocked by PreToolUse: "
                        f"destructive command: {command}"
                    ),
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
        return handler(request)
