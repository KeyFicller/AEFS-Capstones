"""Record executor model/tool order and write a Mermaid sequence diagram."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langchain_core.runnables.graph_mermaid import draw_mermaid_png

SequenceEvent = tuple[str, ...]

_TASK_LABEL_CHARS = 60
_CALL_ARG_CHARS = 120
_BOX_COLORS = (
    "rgb(232, 240, 254)",
    "rgb(232, 245, 233)",
    "rgb(255, 243, 224)",
    "rgb(243, 229, 245)",
)


def _task_label(text: str) -> str:
    compact = " ".join(text.split()).replace(";", ",").replace("#", "")
    if len(compact) <= _TASK_LABEL_CHARS:
        return compact or "task"
    return compact[: _TASK_LABEL_CHARS - 3] + "..."


def _arg_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _call_preview(args: Any) -> str:
    """One short argument preview. A single value is shown bare; extra keys keep their names."""
    if not isinstance(args, dict) or not args:
        raw = ""
    elif len(args) == 1:
        raw = _arg_text(next(iter(args.values())))
    else:
        raw = ", ".join(f"{key}={_arg_text(value)}" for key, value in args.items())
    compact = " ".join(raw.split()).replace(";", ",").replace("#", "")
    if len(compact) <= _CALL_ARG_CHARS:
        return compact
    return compact[: _CALL_ARG_CHARS - 3] + "..."


def _append_message(lines: list[str], event: SequenceEvent) -> None:
    kind = event[0]
    if kind == "call":
        preview = event[2] if len(event) > 2 else ""
        lines.append(f"    Model->>{event[1]}: call({preview})")
    elif kind == "return":
        lines.append(f"    {event[1]}-->>Model: {event[2]}")
    elif kind == "reply":
        lines.append("    Note over Model: reply")


def sequence_diagram(events: Sequence[SequenceEvent]) -> str:
    """Mermaid sequenceDiagram of model/tool order. Call arrows include a short arg preview.

    A ``("task", description)`` event opens a box around the following calls
    until the next task. The description is clipped to one short line.
    """
    tools: list[str] = []
    for event in events:
        if event[0] in {"call", "return"} and event[1] not in tools:
            tools.append(event[1])

    lines = ["sequenceDiagram", "    participant Model"]
    lines.extend(f"    participant {name}" for name in tools)
    span = "Model" if not tools else f"Model,{tools[-1]}"
    box = 0
    open_box = False
    for event in events:
        if event[0] == "task":
            if open_box:
                lines.append("    end")
            color = _BOX_COLORS[box % len(_BOX_COLORS)]
            box += 1
            lines.append(f"    rect {color}")
            lines.append(f"    Note over {span}: {_task_label(event[1] if len(event) > 1 else '')}")
            open_box = True
            continue
        _append_message(lines, event)
    if open_box:
        lines.append("    end")
    return "\n".join(lines) + "\n"


def write_sequence_png(events: Sequence[SequenceEvent] | None, path: Path | None) -> None:
    """Render one sequence diagram for a whole graph invoke. No path means skip."""
    if not path or not events:
        return
    png_path = path if path.suffix.lower() == ".png" else path / "sequence.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.write_bytes(draw_mermaid_png(sequence_diagram(events)))


class SequenceMiddleware(AgentMiddleware):
    """Append to a shared event list. The graph writes the PNG once, at the end.

    Only installed when a sequence log was requested (`sequence_path` in config),
    so `events` is always a real list — there is no "disabled" mode inside.
    """

    def __init__(self, events: list[SequenceEvent], *, description: str = "") -> None:
        super().__init__()
        self.events = events
        if description:
            events.append(("task", description))

    def _note_model_turn(self, state: Any) -> None:
        messages = state.get("messages") or []
        last_ai = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )
        if last_ai is None:
            return
        if last_ai.tool_calls:
            self.events.extend(
                ("call", call["name"], _call_preview(call.get("args")))
                for call in last_ai.tool_calls
            )
            return
        self.events.append(("reply",))

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self._note_model_turn(state)
        return None

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    def wrap_tool_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        name = (request.tool_call or {}).get("name") or "tool"
        try:
            result = handler(request)
        except Exception:
            self.events.append(("return", name, "error"))
            raise
        status = getattr(result, "status", None) or "success"
        self.events.append(("return", name, status))
        return result

    async def awrap_tool_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        name = (request.tool_call or {}).get("name") or "tool"
        try:
            result = await handler(request)
        except Exception:
            self.events.append(("return", name, "error"))
            raise
        status = getattr(result, "status", None) or "success"
        self.events.append(("return", name, status))
        return result
