"""Record-only OpenAI-compatible logging proxy for demos."""

from __future__ import annotations

import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from prompt_augment_lens.normalize import extract_user_query
from prompt_augment_lens.paths import DEFAULT_CAPTURES_DIR

log = logging.getLogger(__name__)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return str(content)


def infer_utterance(body: dict[str, Any]) -> str:
    """Best-effort UI utterance from a chat.completions body."""
    for message in reversed(body.get("messages") or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _message_text(message)
        queried = extract_user_query(text)
        if queried is not None:
            return queried
    for message in reversed(body.get("messages") or []):
        if isinstance(message, dict) and message.get("role") == "user":
            return _message_text(message).strip()
    return ""


def stub_chat_completion(body: dict[str, Any]) -> dict[str, Any]:
    """Return a tiny stub completion so clients do not hang."""
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
            if parsed.path.rstrip("/") not in {
                "/v1/chat/completions",
                "/chat/completions",
            }:
                self._send_json(404, {"error": {"message": "not found"}})
                return

            # Drop auth headers from anything we persist.
            body = self._read_json()
            utterance = infer_utterance(body)
            captures_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            out = captures_dir / f"capture-{stamp}.json"
            record = {
                "vendor": "openai_compatible",
                "visibility": "model_api",
                "user_utterance_raw": utterance,
                "request": body,
            }
            out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info("Recorded capture %s (%s bytes UI text)", out.name, len(utterance))
            self._send_json(200, stub_chat_completion(body))

    return Handler


def serve_proxy(
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    captures_dir: Path | None = None,
) -> None:
    """Serve the record-only chat completions endpoint."""
    directory = captures_dir or DEFAULT_CAPTURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    handler = make_handler(directory)
    server = ThreadingHTTPServer((host, port), handler)
    log.info("Logging proxy on http://%s:%s (captures=%s)", host, port, directory)
    server.serve_forever()
