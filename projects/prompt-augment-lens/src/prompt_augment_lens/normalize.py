"""Normalize OpenAI-compatible chat.completions bodies into AugmentedTurn."""

from __future__ import annotations

import json
import re
from typing import Any

from prompt_augment_lens.models import AugmentedTurn, DiffSummary, Layer, Visibility

_USER_QUERY_RE = re.compile(
    r"<user_query>\s*(.*?)\s*</user_query>",
    re.IGNORECASE | re.DOTALL,
)
_SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>\s*.*?\s*</system-reminder>",
    re.IGNORECASE | re.DOTALL,
)


def extract_user_query(text: str) -> str | None:
    """Return the inner <user_query> text when present."""
    match = _USER_QUERY_RE.search(text or "")
    if not match:
        return None
    return match.group(1).strip()


def strip_system_reminders(text: str) -> str:
    """Remove Claude Code <system-reminder> blocks from a user message."""
    return _SYSTEM_REMINDER_RE.sub("", text or "").strip()


def extract_ui_utterance(text: str) -> str:
    """Best-effort human utterance from a possibly-augmented user message."""
    queried = extract_user_query(text)
    if queried is not None:
        return queried
    cleaned = strip_system_reminders(text)
    return cleaned or (text or "").strip()


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def _classify_user_block(text: str, index: int) -> Layer:
    lower = text.lower()
    # Prefer the utterance marker when a composite user block mixes tags.
    if "<user_query>" in lower:
        return Layer.from_text(f"user_query_{index}", "user_utterance", text)
    if "<system-reminder>" in lower:
        return Layer.from_text(f"system_reminder_{index}", "project_memory", text)
    if "<custom_instructions>" in lower or "<user_rules>" in lower:
        return Layer.from_text(f"user_rules_{index}", "rules_or_instructions", text)
    if "<user_info>" in lower or "<project_layout>" in lower:
        return Layer.from_text(f"environment_{index}", "environment", text)
    if text.lstrip().startswith("# AGENTS.md") or "CLAUDE.md" in text:
        return Layer.from_text(f"project_memory_{index}", "project_memory", text)
    return Layer.from_text(f"user_context_{index}", "file_or_editor_context", text)


def _tools_layer(tools: Any) -> Layer | None:
    if not tools:
        return None
    dumped = json.dumps(tools, ensure_ascii=False, indent=2)
    names: list[str] = []
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function")
            if isinstance(fn, dict) and fn.get("name"):
                names.append(str(fn["name"]))
            elif tool.get("name"):
                names.append(str(tool["name"]))
    label = "tools: " + (", ".join(names) if names else f"{len(tools)} definitions")
    return Layer.from_text(label, "tools", dumped)


def build_diff(
    user_utterance_raw: str,
    user_utterance_in_payload: str,
    layers: list[Layer],
) -> DiffSummary:
    """Summarize augmentation relative to the UI utterance."""
    ui = user_utterance_raw or ""
    in_payload = user_utterance_in_payload or ""
    present_in_layers = any(ui and ui in layer.text for layer in layers)
    unchanged = bool(ui) and (ui == in_payload or present_in_layers)
    wrapped = bool(ui) and any(
        ui in layer.text and layer.text.strip() != ui.strip()
        for layer in layers
        if layer.kind in {"user_utterance", "project_memory"}
    )
    model_bytes = sum(layer.bytes for layer in layers)
    added = [layer.name for layer in layers if layer.kind != "user_utterance"]
    return DiffSummary(
        added_layers=added,
        utterance_unchanged=unchanged,
        utterance_wrapped=wrapped,
        bytes_ui=len(ui.encode("utf-8")),
        bytes_model=model_bytes,
    )


def _system_text(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for part in system:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    return str(system)


def looks_like_anthropic_messages(body: dict[str, Any]) -> bool:
    """Heuristic: Anthropic puts system top-level and tools use input_schema."""
    if "system" in body:
        return True
    tools = body.get("tools") or []
    return any(isinstance(tool, dict) and "input_schema" in tool for tool in tools)


def normalize_chat_completions(
    body: dict[str, Any],
    *,
    user_utterance_raw: str | None = None,
    vendor: str = "openai_compatible",
    visibility: Visibility = "model_api",
    raw_ref: str = "",
) -> AugmentedTurn:
    """Turn a chat.completions or Anthropic messages body into an AugmentedTurn."""
    layers: list[Layer] = []
    utterance_in_payload = ""
    user_index = 0

    system = _system_text(body.get("system"))
    if system:
        layers.append(Layer.from_text("system", "system", system))

    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        text = _message_text(message)
        if role == "system":
            kind = "environment" if text.lstrip().startswith("# Environment") else "system"
            layers.append(Layer.from_text(f"{kind}_{user_index or 'sys'}", kind, text))
            continue
        if role == "developer":
            layers.append(Layer.from_text("developer", "system", text))
            continue
        if role == "user":
            user_index += 1
            layer = _classify_user_block(text, user_index)
            layers.append(layer)
            utterance_in_payload = extract_ui_utterance(text)
            # Keep a dedicated utterance layer when reminders wrap the ask.
            cleaned = strip_system_reminders(text)
            if (
                layer.kind == "project_memory"
                and cleaned
                and cleaned != text.strip()
            ):
                layers.append(
                    Layer.from_text(f"user_utterance_{user_index}", "user_utterance", cleaned)
                )
            continue
        if role == "assistant":
            layers.append(Layer.from_text(f"assistant_{user_index}", "other", text))
            continue
        layers.append(Layer.from_text(f"{role or 'unknown'}_{user_index}", "other", text))

    tools_layer = _tools_layer(body.get("tools"))
    if tools_layer is not None:
        layers.append(tools_layer)

    if not utterance_in_payload:
        # Fall back to the last plain user text block.
        for layer in reversed(layers):
            if layer.kind in {"user_utterance", "file_or_editor_context"}:
                utterance_in_payload = extract_ui_utterance(layer.text)
                break

    inferred = utterance_in_payload
    raw = user_utterance_raw if user_utterance_raw is not None else inferred
    # If the capture stored the whole augmented user blob as raw, peel it.
    if user_utterance_raw and "<system-reminder>" in user_utterance_raw:
        raw = extract_ui_utterance(user_utterance_raw)
    diff = build_diff(raw, utterance_in_payload or raw, layers)
    return AugmentedTurn(
        vendor=vendor,
        visibility=visibility,
        user_utterance_raw=raw,
        user_utterance_in_payload=utterance_in_payload or raw,
        layers=layers,
        diff_summary=diff,
        raw_ref=raw_ref,
        model=str(body.get("model") or ""),
    )


def normalize_capture_file(path: str) -> AugmentedTurn:
    """Load a JSON capture file and normalize it."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if "request" in payload and isinstance(payload["request"], dict):
        body = payload["request"]
        raw = payload.get("user_utterance_raw")
        vendor = str(payload.get("vendor") or "openai_compatible")
        if vendor == "openai_compatible" and looks_like_anthropic_messages(body):
            vendor = "claude_code"
        visibility = payload.get("visibility") or "model_api"
        return normalize_chat_completions(
            body,
            user_utterance_raw=raw if isinstance(raw, str) else None,
            vendor=vendor,
            visibility=visibility,  # type: ignore[arg-type]
            raw_ref=path,
        )

    if "messages" in payload:
        vendor = "claude_code" if looks_like_anthropic_messages(payload) else "openai_compatible"
        return normalize_chat_completions(payload, vendor=vendor, raw_ref=path)

    raise ValueError(f"Unrecognized capture shape in {path}")
