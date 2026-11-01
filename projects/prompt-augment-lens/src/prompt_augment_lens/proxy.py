"""Logging proxy for OpenAI-compatible and Anthropic Messages captures."""

from __future__ import annotations

import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from prompt_augment_lens.normalize import extract_ui_utterance
from prompt_augment_lens.paths import DEFAULT_CAPTURES_DIR

log = logging.getLogger(__name__)


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


def infer_utterance(body: dict[str, Any]) -> str:
    """Best-effort UI utterance from chat.completions or Anthropic messages."""
    for message in reversed(body.get("messages") or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _message_text(message)
        utterance = extract_ui_utterance(text)
        if utterance:
            return utterance
    return ""


def stub_chat_completion(body: dict[str, Any]) -> dict[str, Any]:
    """Return a tiny stub OpenAI chat completion."""
    model = str(body.get("model") or "demo-model")
    return {
        "id": "chatcmpl-prompt-augment-lens-demo",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        "prompt-augment-lens recorded this request. "
                        "Open the compare UI to inspect UI vs model input."
                    ),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def stub_anthropic_message(body: dict[str, Any]) -> dict[str, Any]:
    """Return a tiny stub Anthropic Messages response."""
    model = str(body.get("model") or "claude-demo")
    return {
        "id": "msg_prompt_augment_lens_demo",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "text",
                "text": (
                    "prompt-augment-lens recorded this Claude request. "
                    "Open the compare UI to inspect UI vs model input."
                ),
            }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def anthropic_sse(body: dict[str, Any]) -> bytes:
    """Build a minimal Anthropic SSE stream for streaming clients."""
    message = stub_anthropic_message(body)
    text = message["content"][0]["text"]
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message["id"],
                    "type": "message",
                    "role": "assistant",
                    "model": message["model"],
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        (
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    chunks: list[str] = []
    for event, payload in events:
        chunks.append(f"event: {event}\ndata: {json.dumps(payload)}\n\n")
    return "".join(chunks).encode("utf-8")


def write_capture(
    captures_dir: Path,
    *,
    vendor: str,
    body: dict[str, Any],
    wire: str,
) -> Path:
    """Persist a redacted capture JSON (no auth headers)."""
    captures_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = captures_dir / f"capture-{vendor}-{stamp}.json"
    record = {
        "vendor": vendor,
        "visibility": "model_api",
        "wire": wire,
        "user_utterance_raw": infer_utterance(body),
        "request": body,
    }
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Recorded %s (%s)", out.name, wire)
    return out


def make_handler(captures_dir: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            log.info("%s - %s", self.address_string(), fmt % args)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_bytes(self, code: int, raw: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path.rstrip("/") == "/v1/models":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": "demo-model",
                                "object": "model",
                                "owned_by": "prompt-augment-lens",
                            }
                        ],
                    },
                )
                return
            self._send_json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")
            body = self._read_json()

            if path in {"/v1/chat/completions", "/chat/completions"}:
                write_capture(
                    captures_dir,
                    vendor="openai_compatible",
                    body=body,
                    wire="openai-chat",
                )
                self._send_json(200, stub_chat_completion(body))
                return

            if path in {"/v1/messages", "/messages"}:
                write_capture(
                    captures_dir,
                    vendor="claude_code",
                    body=body,
                    wire="anthropic-messages",
                )
                if body.get("stream"):
                    self._send_bytes(200, anthropic_sse(body), "text/event-stream")
                else:
                    self._send_json(200, stub_anthropic_message(body))
                return

            # Claude Code may probe other endpoints; acknowledge cheaply.
            if path.endswith(":count_tokens") or path.endswith("/count_tokens"):
                self._send_json(200, {"input_tokens": 1})
                return

            self._send_json(404, {"error": {"message": f"not found: {path}"}})

    return Handler


def serve_proxy(
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    captures_dir: Path | None = None,
) -> None:
    """Serve the logging proxy until interrupted."""
    directory = captures_dir or DEFAULT_CAPTURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    handler = make_handler(directory)
    server = ThreadingHTTPServer((host, port), handler)
    log.info("Logging proxy on http://%s:%s (captures=%s)", host, port, directory)
    server.serve_forever()
