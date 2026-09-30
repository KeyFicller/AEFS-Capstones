from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from terminal_coding_agent.telemetry.setup import get_tracer, setup_tracing

from pathlib import Path

def test_setup_tracing_records_spans(tmp_path: Path):
    exporter = InMemorySpanExporter()
    setup_tracing(worktree=tmp_path, exporter=exporter)
    tracer = get_tracer()
    with tracer.start_as_current_span("probe"):
        pass
    span = exporter.get_finished_spans()
    assert any(s.name == "probe" for s in span)