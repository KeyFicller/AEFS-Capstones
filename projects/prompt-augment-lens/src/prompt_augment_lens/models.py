"""Normalized capture of a UI utterance versus model-bound input."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Visibility = Literal["model_api", "fixture", "cursor_backend"]


@dataclass(frozen=True)
class Layer:
    """One augmentation layer peeled from the outbound payload."""

    name: str
    kind: str
    text: str
    bytes: int

    @classmethod
    def from_text(cls, name: str, kind: str, text: str) -> Layer:
        body = text or ""
        return cls(name=name, kind=kind, text=body, bytes=len(body.encode("utf-8")))


@dataclass(frozen=True)
class DiffSummary:
    """How the model-bound input differs from the UI utterance."""

    added_layers: list[str]
    utterance_unchanged: bool
    utterance_wrapped: bool
    bytes_ui: int
    bytes_model: int


@dataclass
class AugmentedTurn:
    """One captured turn ready for the compare UI."""

    vendor: str
    visibility: Visibility
    user_utterance_raw: str
    user_utterance_in_payload: str
    layers: list[Layer] = field(default_factory=list)
    diff_summary: DiffSummary | None = None
    raw_ref: str = ""
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload
