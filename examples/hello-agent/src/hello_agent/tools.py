"""工具层：唯一允许产生副作用的边界。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str


def say_hello(name: str) -> ToolResult:
    return ToolResult(ok=True, output=f"hello, {name}!")


TOOLS: dict[str, Callable[[str], ToolResult]] = {"say_hello": say_hello}
