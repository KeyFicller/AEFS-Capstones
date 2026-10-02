"""Langfuse LangChain CallbackHandler wiring (framework integration).

Prefer this over raw OTLP for LangGraph/LangChain so observation types, nesting,
and token/model fields follow Langfuse best practices.
Docs: https://langfuse.com/integrations/frameworks/langchain
"""

from typing import Any

from terminal_coding_agent.telemetry.setup import langfuse_env


def langfuse_callback_handler() -> Any | None:
    """Return a Langfuse CallbackHandler when keys are set; else None (offline/tests)."""
    if langfuse_env() is None:
        return None
    # SDK also reads LANGFUSE_HOST; keep it aligned with BASE_URL (skill/CLI note).
    import os

    from terminal_coding_agent.telemetry.setup import DEFAULT_LANGFUSE_BASE_URL

    base = os.environ.get("LANGFUSE_BASE_URL", "").strip() or DEFAULT_LANGFUSE_BASE_URL
    os.environ["LANGFUSE_BASE_URL"] = base
    os.environ.setdefault("LANGFUSE_HOST", base)
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")

    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def flush_langfuse() -> None:
    """Flush pending Langfuse spans (scripts / __main__)."""
    if langfuse_env() is None:
        return
    from langfuse import get_client

    get_client().flush()
