"""Minimal local compare UI over stdlib HTTP."""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from prompt_augment_lens.normalize import normalize_capture_file
from prompt_augment_lens.paths import DEFAULT_CAPTURES_DIR, FIXTURES_DIR, STATIC_DIR

log = logging.getLogger(__name__)


def _load_turn(path: Path) -> dict:
    turn = normalize_capture_file(str(path))
    return turn.to_dict()


def _resolve_default_turn(captures_dir: Path) -> Path:
    captures = sorted(captures_dir.glob("*.json"))
    if captures:
        return captures[-1]
    fixture = FIXTURES_DIR / "sample_agent_turn.json"
    if not fixture.is_file():
        raise FileNotFoundError(f"Missing fixture: {fixture}")
    return fixture


def make_handler(turn_path: Path, captures_dir: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            log.info("%s - %s", self.address_string(), fmt % args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                index = STATIC_DIR / "index.html"
                self._send(200, index.read_bytes(), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/turn":
                path = turn_path if turn_path.is_file() else _resolve_default_turn(captures_dir)
                payload = {
                    "path": str(path),
                    "turn": _load_turn(path),
                }
                raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
                self._send(200, raw, "application/json; charset=utf-8")
                return
            if parsed.path.startswith("/static/"):
                rel = parsed.path.removeprefix("/static/")
                file_path = (STATIC_DIR / rel).resolve()
                if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.is_file():
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                content_type = "text/css; charset=utf-8" if file_path.suffix == ".css" else "text/plain"
                if file_path.suffix == ".js":
                    content_type = "application/javascript; charset=utf-8"
                self._send(200, file_path.read_bytes(), content_type)
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

    return Handler


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    turn_path: Path | None = None,
    captures_dir: Path | None = None,
) -> None:
    """Serve the compare UI until interrupted."""
    captures = captures_dir or DEFAULT_CAPTURES_DIR
    captures.mkdir(parents=True, exist_ok=True)
    path = turn_path or _resolve_default_turn(captures)
    handler = make_handler(path, captures)
    server = ThreadingHTTPServer((host, port), handler)
    log.info("Compare UI on http://%s:%s (turn=%s)", host, port, path)
    server.serve_forever()
