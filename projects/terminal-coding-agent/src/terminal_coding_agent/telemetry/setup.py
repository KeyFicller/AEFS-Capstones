import base64
import json
import os
from pathlib import Path

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Tracer

# Relative path under a worktree (string form for callers that concatenate).
OTEL_JSONL_RELPATH = ".agent/otel.jsonl"
DEFAULT_LANGFUSE_BASE_URL = "https://cloud.langfuse.com"


def otel_jsonl_path(worktree: Path | str) -> Path:
    """Build `{worktree}/.agent/otel.jsonl` via string concat (stable for copy/export)."""
    return Path(str(worktree).rstrip("/") + "/" + OTEL_JSONL_RELPATH)


def langfuse_env() -> tuple[str, str, str] | None:
    """Return (public_key, secret_key, base_url) when both keys are set; else None."""
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    if not public_key or not secret_key:
        return None
    base_url = os.environ.get("LANGFUSE_BASE_URL", "").strip() or DEFAULT_LANGFUSE_BASE_URL
    return public_key, secret_key, base_url


def langfuse_otlp_exporter(*, public_key: str, secret_key: str, base_url: str) -> SpanExporter:
    """OTLP HTTP exporter aimed at Langfuse `/api/public/otel`."""
    endpoint = f"{base_url.rstrip('/')}/api/public/otel"
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return OTLPSpanExporter(
        endpoint=endpoint,
        headers={
            "Authorization": f"Basic {token}",
            "x-langfuse-ingestion-version": "4",
        },
    )


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


def configure_default_exporters(provider: TracerProvider, worktree: Path) -> None:
    """JSONL always. Optional OTLP only when LANGFUSE_OTLP=1 (CallbackHandler is preferred)."""
    provider.add_span_processor(SimpleSpanProcessor(JSONLSpanExporter(otel_jsonl_path(worktree))))
    if os.environ.get("LANGFUSE_OTLP", "").strip() != "1":
        return
    creds = langfuse_env()
    if creds is None:
        return
    public_key, secret_key, base_url = creds
    provider.add_span_processor(
        BatchSpanProcessor(
            langfuse_otlp_exporter(
                public_key=public_key,
                secret_key=secret_key,
                base_url=base_url,
            )
        )
    )


def setup_tracing(*, worktree: Path, exporter: SpanExporter | None = None) -> TracerProvider:
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        # Global provider is once-only; attach extra exporters for later callers (tests).
        if exporter is not None:
            current.add_span_processor(SimpleSpanProcessor(exporter))
        return current

    provider = TracerProvider()
    if exporter is None:
        configure_default_exporters(provider, worktree)
    else:
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    return provider


def get_tracer(name: str = "terminal_coding_agent") -> Tracer:
    return trace.get_tracer(name)
