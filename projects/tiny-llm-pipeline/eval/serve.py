"""A local page for one question and three stages, side by side.

Stdlib only: `http.server` plus a single self-contained page. Nothing is
written to disk, so a reload starts over; archive a run with
`eval/sample.py --html` instead.

`pretrain` is a base model, so its column is a raw continuation rather than an
answer. The page says so on that card.

Inference runs on the main thread of a single-threaded `HTTPServer`: MPS is not
thread-safe, and one request at a time is plenty for a page you ask questions
of by hand.

Run it directly:

    python eval/serve.py [--bundle <dir>] [--port 8000] [--open]

`eval/` is prepended to `sys.path` when this runs as a script, so it must not
contain a file named after a stdlib or third-party module.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import time
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from string import Template

from tiny_llm_pipeline.generate import Bundle, GenSettings

BUNDLE = Path(__file__).resolve().parents[1] / "checkpoints"
MAX_NEW = 512
TEMPERATURE_MAX = 2.0
REP_PEN_MIN, REP_PEN_MAX = 1.0, 2.0

logger = logging.getLogger("serve")

# `string.Template`, not `str.format`: the CSS and JS below are full of braces.
# Keep `$` out of the page, since every `$` is a placeholder.
_PAGE = Template(
    """<!doctype html>
<meta charset="utf-8">
<title>pretrain / sft / dpo</title>
<style>
  body { font: 15px/1.6 -apple-system, "PingFang SC", sans-serif; margin: 32px auto;
         max-width: 1280px; color: #1a1a1a; background: #fafafa; }
  h1 { font-size: 20px; }
  .meta { color: #666; font-size: 13px; }
  textarea { width: 100%; box-sizing: border-box; font: inherit; padding: 8px;
             border: 1px solid #ccc; border-radius: 6px; background: #fff; }
  .controls { display: flex; gap: 16px; align-items: center; margin: 10px 0 4px; }
  .controls label { color: #555; font-size: 13px; }
  .controls input { width: 72px; font: inherit; padding: 4px 6px;
                    border: 1px solid #ccc; border-radius: 4px; }
  button { font: inherit; padding: 6px 18px; border: 0; border-radius: 6px;
           background: #0b5cad; color: #fff; cursor: pointer; }
  button:disabled { background: #9bb6d0; cursor: default; }
  .error { color: #b00020; font-size: 13px; min-height: 18px; margin: 8px 0 0; }
  .cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 16px; }
  .card { background: #fff; border: 1px solid #e3e3e3; border-radius: 6px; padding: 12px; }
  .card h3 { margin: 0 0 6px; font-size: 13px; color: #0b5cad; font-weight: 600; }
  .note { color: #999; font-size: 12px; margin: 0 0 6px; }
  .stats { color: #999; font-size: 12px; margin: 8px 0 2px; }
  pre { margin: 0; white-space: pre-wrap; word-wrap: break-word; font: inherit; }
</style>
<h1>pretrain / sft / dpo — one question, three stages</h1>
<p class="meta">device=$device · tokenizer=$tokenizer · $header</p>

<form id="ask">
  <textarea id="question" rows="3" placeholder="问一个问题…"></textarea>
  <div class="controls">
    <label>rep_pen <input id="rep_pen" type="number" step="0.1" min="1" max="2" value="1.5"></label>
    <label>temperature <input id="temperature" type="number" step="0.1" min="0" max="2" value="0"></label>
    <label>max_new <input id="max_new" type="number" step="1" min="1" max="512" value="64"></label>
    <button id="go" type="submit">生成</button>
  </div>
</form>
<p id="error" class="error"></p>
<p id="timing" class="meta"></p>
<div class="cards" id="cards"></div>

<script>
const STAGES = $stages;
const NOTE = "base model: this column continues the text, it does not answer.";
const cards = document.getElementById("cards");
const error = document.getElementById("error");
const timing = document.getElementById("timing");
const go = document.getElementById("go");

function numberInput(id) {
  const value = Number(document.getElementById(id).value);
  return Number.isFinite(value) ? value : undefined;
}

function card(stage, data) {
  const box = document.createElement("div");
  box.className = "card";
  const heading = document.createElement("h3");
  heading.textContent = stage + " (step " + data.step + ")";
  box.appendChild(heading);
  if (stage === "pretrain") {
    const note = document.createElement("p");
    note.className = "note";
    note.textContent = NOTE;
    box.appendChild(note);
  }
  const reading = document.createElement("div");
  reading.className = "stats";
  reading.textContent = "chars=" + data.stats.chars +
    " dist4=" + data.stats.dist4.toFixed(2) + " tempo=" + data.stats.tempo;
  box.appendChild(reading);
  const reply = document.createElement("pre");
  reply.textContent = data.reply || "(empty)";
  box.appendChild(reply);
  return box;
}

document.getElementById("ask").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    question: document.getElementById("question").value,
    rep_pen: numberInput("rep_pen"),
    temperature: numberInput("temperature"),
    max_new: numberInput("max_new"),
  };
  error.textContent = "";
  go.disabled = true;
  const label = go.textContent;
  go.textContent = "生成中…";
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      error.textContent = data.error || "request failed";
      return;
    }
    const rendered = [];
    for (const stage of STAGES) {
      const found = data.stages.find((item) => item.stage === stage);
      if (found) rendered.push(card(stage, found));
    }
    cards.replaceChildren(...rendered);
    timing.textContent = "elapsed " + data.elapsed_ms + " ms";
  } catch (failure) {
    error.textContent = String(failure);
  } finally {
    go.disabled = false;
    go.textContent = label;
  }
});
</script>
"""
)


def _number(value: object, default: float, low: float, high: float) -> float:
    """`value` clamped to `[low, high]`, or `default` when it is not a number."""
    try:
        return min(max(float(value), low), high)
    except (TypeError, ValueError):
        return default


def settings_from(payload: dict) -> GenSettings:
    """`GenSettings` for a request. The page exposes these three numbers only.

    Out-of-range and unparsable values clamp instead of raising: a typo in the
    page must not take the server down or come back as a 500.
    """
    defaults = GenSettings()
    return GenSettings(
        max_new=int(_number(payload.get("max_new"), defaults.max_new, 1, MAX_NEW)),
        temperature=_number(
            payload.get("temperature"), defaults.temperature, 0.0, TEMPERATURE_MAX
        ),
        rep_pen=_number(payload.get("rep_pen"), defaults.rep_pen, REP_PEN_MIN, REP_PEN_MAX),
    )


def render_page(bundle: Bundle) -> str:
    """The page, with the run's provenance stamped into the header."""
    header = " · ".join(f"{stage} step {step}" for stage, step in bundle.steps.items())
    return _PAGE.substitute(
        device=bundle.device.type,
        tokenizer=bundle.tok.hash,
        header=html.escape(header),
        stages=json.dumps(list(bundle.stages)),
    )


def handle_ask(bundle: Bundle, payload: object) -> tuple[int, dict]:
    """`(status, body)` for a `/ask` payload. Writes nothing to disk."""
    if not isinstance(payload, dict):
        return 400, {"error": "request body must be a JSON object"}
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return 400, {"error": "question must be a non-empty string"}
    started = time.perf_counter()
    try:
        replies = bundle.run(question.strip(), settings_from(payload))
    except Exception as exc:                 # noqa: BLE001 - reported, not swallowed
        logger.exception("generation failed")
        return 500, {"error": f"{type(exc).__name__}: {exc}"}
    elapsed = int((time.perf_counter() - started) * 1000)
    return 200, {
        "device": bundle.device.type,
        "elapsed_ms": elapsed,
        "stages": [asdict(reply) for reply in replies],
    }


def make_handler(bundle: Bundle) -> type[BaseHTTPRequestHandler]:
    """A request handler bound to one loaded bundle."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:            # noqa: N802 - stdlib naming
            if self._route() != "/":
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            self._send(
                200, render_page(bundle).encode("utf-8"), "text/html; charset=utf-8"
            )

        def do_POST(self) -> None:           # noqa: N802 - stdlib naming
            if self._route() != "/ask":
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw or b"null")
            except json.JSONDecodeError:
                status, body = 400, {"error": "request body is not valid JSON"}
            else:
                status, body = handle_ask(bundle, payload)
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self._send(status, encoded, "application/json; charset=utf-8")

        def _route(self) -> str:
            return self.path.split("?", 1)[0]

        def _send(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args: object) -> None:
            logger.info("%s %s", self.address_string(), fmt % args)

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Local three-stage reply page.")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bundle = Bundle.load(args.bundle)
    steps = " ".join(f"{stage}={step}" for stage, step in bundle.steps.items())
    print(f"device={bundle.device.type} tokenizer={bundle.tok.hash} {steps}")
    server = HTTPServer(("127.0.0.1", args.port), make_handler(bundle))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"serving {url} (ctrl-c to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
