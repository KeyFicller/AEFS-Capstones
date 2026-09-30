from __future__ import annotations

import json
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Tracer

# Relative path under a worktree (string form for callers that concatenate).
OTEL_JSONL_RELPATH = ".agent/otel.jsonl"


def otel_jsonl_path(worktree: Path | str) -> Path:
    """Build `{worktree}/.agent/otel.jsonl` via string concat (stable for copy/export)."""
    return Path(str(worktree).rstrip("/") + "/" + OTEL_JSONL_RELPATH)


class JSONLSpanExporter(SpanExporter):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = None

    def export(self, spans) -> SpanExportResult:
        if self.file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(self.path, "a", encoding="utf-8")
        for span in spans:
            self.file.write(json.dumps(json.loads(span.to_json())) + "\n")
            self.file.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


def setup_tracing(
    *, worktree: Path, exporter: SpanExporter | None = None
) -> TracerProvider:
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        # Global provider is once-only; attach extra exporters for later callers (tests).
        if exporter is not None:
            current.add_span_processor(SimpleSpanProcessor(exporter))
        return current

    provider = TracerProvider()
    if exporter is None:
        # Local default: JSONL only (Console is too noisy for demo/CLI).
        provider.add_span_processor(
            SimpleSpanProcessor(JSONLSpanExporter(otel_jsonl_path(worktree)))
        )
    else:
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    return provider


def get_tracer(name: str = "terminal_coding_agent") -> Tracer:
    return trace.get_tracer(name)
