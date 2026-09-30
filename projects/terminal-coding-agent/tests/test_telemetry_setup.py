from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from terminal_coding_agent.telemetry.langfuse_callback import langfuse_callback_handler
from terminal_coding_agent.telemetry.setup import (
    configure_default_exporters,
    get_tracer,
    langfuse_env,
    langfuse_otlp_exporter,
    setup_tracing,
)


def test_setup_tracing_records_spans(tmp_path: Path) -> None:
    exporter = InMemorySpanExporter()
    setup_tracing(worktree=tmp_path, exporter=exporter)
    tracer = get_tracer()
    with tracer.start_as_current_span("probe"):
        pass
    spans = exporter.get_finished_spans()
    assert any(s.name == "probe" for s in spans)


def test_langfuse_env_none_without_keys(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert langfuse_env() is None


def test_langfuse_env_reads_keys_and_base(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:3000")
    assert langfuse_env() == ("pk-test", "sk-test", "http://localhost:3000")


def test_langfuse_otlp_exporter_endpoint_and_headers() -> None:
    with patch(
        "terminal_coding_agent.telemetry.setup.OTLPSpanExporter"
    ) as exporter_cls:
        exporter_cls.return_value = MagicMock(name="otlp")
        langfuse_otlp_exporter(
            public_key="pk-test",
            secret_key="sk-test",
            base_url="http://localhost:3000/",
        )
        kwargs = exporter_cls.call_args.kwargs
        assert kwargs["endpoint"] == "http://localhost:3000/api/public/otel"
        expected = base64.b64encode(b"pk-test:sk-test").decode()
        assert kwargs["headers"]["Authorization"] == f"Basic {expected}"
        assert kwargs["headers"]["x-langfuse-ingestion-version"] == "4"


def test_configure_default_exporters_jsonl_only_even_with_keys(
    tmp_path: Path, monkeypatch
) -> None:
    """CallbackHandler is the default Langfuse path; OTLP needs LANGFUSE_OTLP=1."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.delenv("LANGFUSE_OTLP", raising=False)
    provider = TracerProvider()
    configure_default_exporters(provider, tmp_path)
    processors = list(provider._active_span_processor._span_processors)
    assert len(processors) == 1
    assert isinstance(processors[0], SimpleSpanProcessor)


def test_configure_default_exporters_otlp_opt_in(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_OTLP", "1")
    with patch(
        "terminal_coding_agent.telemetry.setup.langfuse_otlp_exporter",
        return_value=MagicMock(name="otlp"),
    ):
        provider = TracerProvider()
        configure_default_exporters(provider, tmp_path)
    processors = list(provider._active_span_processor._span_processors)
    assert len(processors) == 2
    assert isinstance(processors[0], SimpleSpanProcessor)
    assert isinstance(processors[1], BatchSpanProcessor)


def test_langfuse_callback_handler_none_without_keys(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert langfuse_callback_handler() is None


def test_langfuse_callback_handler_when_keys_set(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    fake = MagicMock(name="handler")
    with patch(
        "langfuse.langchain.CallbackHandler", return_value=fake
    ) as handler_cls:
        out = langfuse_callback_handler()
    assert out is fake
    handler_cls.assert_called_once_with()
