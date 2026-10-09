"""Tool-side display: the framework prints the header, the tool supplies the body.

`tool_output` is applied *under* `@tool`, which needs a plain function — so the tool name
comes from `fn.__name__` and no tool-name table exists anywhere.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any

MAX_TARGET_CHARS = 60

# off: silent. simple: one line per call. detail: the summary line plus the body.
# Off by default: Harbor imports the tools and must not gain console output.
LEVEL = "off"
LEVELS = ("off", "simple", "detail")


def _is_error(result: Any) -> bool:
    return isinstance(result, str) and result.lstrip().startswith("Error:")


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else "Error"


def _clip(text: str) -> str:
    return text if len(text) <= MAX_TARGET_CHARS else text[: MAX_TARGET_CHARS - 1] + "…"


def _target(params: dict[str, Any]) -> str:
    """The first argument, whatever its name: that is what identifies the call."""
    for value in params.values():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            text = " ".join(str(part) for part in value)
        else:
            text = str(value)
        return _clip(text.replace("\n", " ⏎ "))
    return ""


def exit_code_and_output(result: str, *, line_count: bool = False) -> tuple[str, str]:
    """`(summary, body)` for a tool returning `exit_code: N / stdout: / stderr:`."""
    stdout = result.partition("stdout:\n")[2].partition("stderr:\n")[0].strip("\n")
    stderr = result.partition("stderr:\n")[2].strip("\n")
    exit_code = (
        result.partition("exit_code:")[2].strip().split()[0] if "exit_code:" in result else "?"
    )
    summary = f"exit {exit_code}"
    if line_count:
        lines = len(stdout.splitlines())
        summary += f" · {lines} line" + ("" if lines == 1 else "s")
    body = "\n\n".join(f"```text\n{text}\n```" for text in (stdout, stderr) if text)
    return summary, body


def tool_output(hook: Callable[[dict[str, Any], Any], tuple[str, str]] | None = None):
    """Show one tool call: a `⚙ name target` header, the hook's summary and its body.

    `hook(params, result)` returns `(summary, body)`. The summary is what the call
    produced (`exit 0 · 12 lines`); the body is the markdown detail. `simple` prints
    only the summary, `detail` prints both. The hook runs only on success — an
    `Error:` result is the framework's to render — and a raising hook must not break
    the tool, so its output is dropped.
    """

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            if LEVEL != "off":
                params = dict(inspect.signature(fn).bind(*args, **kwargs).arguments)
                error = _first_line(result) if _is_error(result) else None
                summary = ""
                body = ""
                if error is None and hook is not None:
                    try:
                        summary, body = hook(params, result) or ("", "")
                    except Exception:  # noqa: BLE001 - display is a side channel
                        summary, body = "", ""
                # Imported here so importing the tools never pulls in the console layer.
                from terminal_coding_agent import ui

                ui.render_tool_call(
                    name=fn.__name__,
                    target=_target(params),
                    summary=summary or "",
                    body=body or "",
                    error=error,
                    header_only=LEVEL == "simple",
                )
            return result

        return wrapper

    return decorate
