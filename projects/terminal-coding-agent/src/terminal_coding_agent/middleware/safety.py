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
        # Pipe-to-interpreter covers `sudo bash`, `/bin/sh`, `zsh`, `python3`, ...
        r"\b(?:curl|wget)\b[^|\n]*\|\s*(?:sudo\s+)?(?:\S*/)?"
        r"(?:sh|bash|zsh|dash|ksh|python3?|perl|ruby|node)\b",
        r":\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?",  # fork bomb
    )
)

# Quotes and escapes hide a payload from the patterns above; the guard also scans a
# version with those characters removed (`sh -c 'rm -rf /'` -> `sh -c rm -rf /`).
_QUOTE_ESCAPE_CHARS = str.maketrans("", "", "'\"\\")


def is_destructive_shell(command: str) -> bool:
    """Return True if the shell command matches the D3 denylist."""
    text = command.strip()
    if not text:
        return False
    for variant in (text, text.translate(_QUOTE_ESCAPE_CHARS)):
        compact = re.sub(r"\s+", "", variant)
        if re.search(r":\(\)\{:\|:&\};?", compact):
            return True
        if any(pattern.search(variant) for pattern in _DESTRUCTIVE_PATTERNS):
            return True
    return False


class SafetyMiddleware(AgentMiddleware):
    """PreToolUse guard: block destructive run_shell commands (D3).

    Both hooks are implemented: the executor calls `agent.invoke`, but
    `AgentMiddleware.awrap_tool_call` *raises* when only the sync hook exists, so
    an async caller (or a future switch to `ainvoke`) would turn the guard into a
    crash instead of a block.
    """

    def _blocked(self, request: Any) -> ToolMessage | None:
        tool_call = getattr(request, "tool_call", None) or {}
        if tool_call.get("name") != "run_shell":
            return None
        command = str((tool_call.get("args") or {}).get("command", ""))
        if not is_destructive_shell(command):
            return None
        return ToolMessage(
            content=(f"Error: blocked by PreToolUse: destructive command: {command}"),
            tool_call_id=tool_call["id"],
            status="error",
        )

    def wrap_tool_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        blocked = self._blocked(request)
        return blocked if blocked is not None else handler(request)

    async def awrap_tool_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        blocked = self._blocked(request)
        return blocked if blocked is not None else await handler(request)
