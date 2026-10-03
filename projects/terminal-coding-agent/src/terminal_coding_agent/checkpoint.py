"""Durable LangGraph checkpointer: {worktree}/.agent/checkpoints.sqlite."""

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

CHECKPOINT_RELPATH = ".agent/checkpoints.sqlite"

# msgpack cannot infer these two app types; allowlist them explicitly.
_ALLOWED_MSGPACK_MODULES = [
    ("terminal_coding_agent.state", "ToDoItem"),
    ("terminal_coding_agent.state", "ToDoStatus"),
]


def checkpoint_path(worktree: Path) -> Path:
    """Where the checkpointer lives: {worktree}/.agent/checkpoints.sqlite."""
    return worktree / CHECKPOINT_RELPATH


def state_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)


class AsyncCapableSqliteSaver(SqliteSaver):
    """`SqliteSaver` that also serves LangGraph's async runtime.

    Harbor's LangGraph runner always calls `ainvoke`, but `SqliteSaver` raises
    `NotImplementedError` for every async method, and `AsyncSqliteSaver` cannot be
    substituted because it builds itself with `asyncio.get_running_loop()` (so the
    sync CLI/tests could not construct it). Delegating to the sync implementation on
    a worker thread keeps one saver usable from both `invoke` and `ainvoke`; the
    connector is already `check_same_thread=False` and `SqliteSaver` holds its own lock.
    """

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        rows: list[CheckpointTuple] = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for row in rows:
            yield row

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)


def build_checkpointer(worktree: Path) -> AsyncCapableSqliteSaver:
    path = checkpoint_path(worktree)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return AsyncCapableSqliteSaver(conn, serde=state_serde())
