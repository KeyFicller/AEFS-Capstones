"""Run one coroutine from sync code on a loop that outlives the call.

``asyncio.run`` per call is not enough: LangChain's async clients cache connections on the
loop they first ran on, so a second ``asyncio.run`` raises ``RuntimeError: Event loop is
closed``. That is why the whole process shares one loop, owned by a daemon thread.

The graph stays synchronous (``graph.invoke`` is what every caller uses), so this is the
seam where concurrency is bought without turning the nodes async.
"""

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")

_LOCK = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None


def _worker_loop() -> asyncio.AbstractEventLoop:
    """Start the shared loop on first use and return it."""
    global _loop, _thread
    with _LOCK:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            _thread = threading.Thread(target=_loop.run_forever, name="mdq-async", daemon=True)
            _thread.start()
        return _loop


def run(coro: Coroutine[Any, Any, _T]) -> _T:
    """Block the calling thread until ``coro`` finishes on the shared loop."""
    if _thread is not None and threading.current_thread() is _thread:
        raise RuntimeError("run() cannot be called from the shared loop's own thread")
    return asyncio.run_coroutine_threadsafe(coro, _worker_loop()).result()
