"""Front-gate chat/work classification. One structured call; no graph, no state."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

Intent = Literal["chat", "work"]

DEFAULT_WINDOW = 3
DEFAULT_PER_TURN_CHARS = 200

_TURN_LABELS = {"human": "User", "ai": "Assistant"}


class IntentVerdict(BaseModel):
    """The classifier's structured output. The field has no default (strict schemas)."""

    intent: Intent = Field(description="chat when a direct reply suffices, else work")


SYSTEM_PROMPT = (
    'Classify one user turn as "chat" or "work".\n\n'
    '- "chat": a direct reply is the deliverable. Greetings, thanks, meta talk, and also '
    "writing / explaining / translating / drafting / summarizing text, or showing content "
    "that already exists. No retrieval or plan step is needed.\n"
    '- "work": the main pipeline is required to deliver, because {work_hint}.\n'
    '- If unsure, choose "work": a misclassified "chat" skips work that was needed.'
)


@dataclass(frozen=True)
class Classification:
    """The verdict plus the model message it came from, so callers can account its usage."""

    intent: Intent
    raw: Any


def _text(content: Any) -> str:
    """Plain text of a message's content: a str, or the text parts of a block list."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    return "\n".join(parts)


def _render(messages: Sequence[BaseMessage], window: int, per_turn_chars: int) -> str:
    """The trailing `window` turns that carry text a human wrote, as a labeled transcript.

    Tool turns and structured parts (images, tool calls) carry no human text and are
    skipped, so they can never reach the structured-output call or consume the window.
    """
    turns: list[tuple[str, str]] = []
    for message in messages:
        label = _TURN_LABELS.get(message.type)
        text = _text(message.content).strip()
        if label and text:
            turns.append((label, text))
    lines = []
    for label, text in turns[-window:]:
        if len(text) > per_turn_chars:
            text = text[:per_turn_chars] + "..."
        lines.append(f"{label}: {text}")
    return "\n\n".join(lines)


class Classifier:
    """Binds one structured call. The model is injected; the caller owns its lifecycle."""

    def __init__(
        self,
        model: BaseChatModel,
        *,
        work_hint: str,
        window: int = DEFAULT_WINDOW,
        per_turn_chars: int = DEFAULT_PER_TURN_CHARS,
    ) -> None:
        self._prompt = SYSTEM_PROMPT.format(work_hint=work_hint)
        self._structured = model.with_structured_output(IntentVerdict, include_raw=True)
        self._window = window
        self._per_turn_chars = per_turn_chars

    def classify(self, messages: Sequence[BaseMessage]) -> Classification:
        """One structured call. Raises on an empty input, model or parse failure."""
        payload = _render(messages, self._window, self._per_turn_chars)
        if not payload:
            raise ValueError("no human-authored turn to classify")
        response = self._structured.invoke(
            [SystemMessage(content=self._prompt), HumanMessage(content=payload)]
        )
        parsed = response["parsed"]
        if parsed is None:
            raise ValueError(f"intent parse failed: {response['parsing_error']}")
        return Classification(intent=parsed.intent, raw=response["raw"])
