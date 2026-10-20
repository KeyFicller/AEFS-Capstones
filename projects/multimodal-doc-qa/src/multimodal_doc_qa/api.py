"""HTTP front end over the same agentic loop the CLI runs.

One process, one encoder load, many concurrent turns. Different conversations run in
parallel; a single conversation runs one turn at a time so its history cannot be lost.
Sessions live in memory, so this is a single-worker deploy: restart clears them, and a
second worker would hold a second, disjoint set.
"""

from __future__ import annotations

import io
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.responses import Response
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError

from multimodal_doc_qa.cli import load_deps
from multimodal_doc_qa.config import ENV_PATH, Settings, load_local_env
from multimodal_doc_qa.graph import build_graph, initial_state
from multimodal_doc_qa.limits import Budget
from multimodal_doc_qa.schemas import Answer, BBox, Citation
from multimodal_doc_qa.ui.viewer import CITED, draw_citations, page_image


@dataclass
class Session:
    """One conversation: the turns so far, and the lock that keeps them in order."""

    history: list[BaseMessage] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = 0.0


class SessionStore:
    """In-process conversation history.

    ``_guard`` covers the dict only. A turn holds its own session's lock across
    ``read history -> invoke -> append``, so two turns of one conversation cannot
    interleave; different conversations never wait on each other.
    """

    def __init__(
        self,
        ttl: float,
        capacity: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl
        self._capacity = capacity
        self._clock = clock
        self._guard = threading.Lock()
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str | None) -> tuple[str, Session]:
        """The session to use, created when absent. An unknown id starts a fresh one.

        Starting fresh rather than 404 keeps clients working across a restart, which
        this in-memory store cannot avoid; the cost is that a mistyped id silently
        opens a new conversation.
        """
        with self._guard:
            now = self._clock()
            self._expire(now)
            chosen = session_id or uuid.uuid4().hex
            session = self._sessions.get(chosen)
            if session is None:
                session = Session(last_used=now)
                self._sessions[chosen] = session
                self._evict_overflow()
            else:
                session.last_used = now
            return chosen, session

    def __len__(self) -> int:
        with self._guard:
            return len(self._sessions)

    def _expire(self, now: float) -> None:
        """Drop idle sessions. Swept on creation, so no background thread is needed."""
        stale = [key for key, session in self._sessions.items() if now - session.last_used > self._ttl]
        for key in stale:
            del self._sessions[key]

    def _evict_overflow(self) -> None:
        while len(self._sessions) > self._capacity:
            oldest = min(self._sessions, key=lambda key: self._sessions[key].last_used)
            del self._sessions[oldest]


@dataclass
class Service:
    """What is loaded once at startup. Everything request-scoped is built per request."""

    settings: Settings
    retriever: object
    mode: str
    artifacts: Path
    sessions: SessionStore


def build_service(settings: Settings) -> Service:
    """Load the retriever now. A missing index fails startup, not the first ask."""
    from multimodal_doc_qa.retrievers.assembly import build_retriever

    return Service(
        settings=settings,
        retriever=build_retriever(settings, settings.mode),
        mode=settings.mode,
        artifacts=settings.artifacts_dir,
        sessions=SessionStore(settings.session_ttl_seconds, settings.max_sessions),
    )


class AskRequest(BaseModel):
    """One turn. ``session_id`` continues a conversation; omitting it starts a new one."""

    question: str = Field(min_length=1)
    session_id: str | None = None


class AskResponse(BaseModel):
    """The answer plus what the ask cost. ``answer`` is ``None`` when the loop found nothing."""

    session_id: str
    answer: Answer | None
    stop_reason: str
    rounds: int
    calls: int
    tokens: int
    mode: str
    latency_s: float


def answer_question(service: Service, question: str, session_id: str | None = None) -> AskResponse:
    """Run one turn of a conversation.

    The budget is per turn: it is attached to the graph as a callback and spent across
    nodes, so a reused one would charge this ask for the previous ask's calls.

    History is read and extended under the session's lock. Everything else -- the models,
    the graph, the state -- is built per request, so concurrent conversations share
    nothing but the retriever.
    """
    started = time.monotonic()
    chosen, session = service.sessions.get_or_create(session_id)
    budget = Budget.from_settings(service.settings)
    deps = load_deps(service.settings, service.retriever, budget, enable_intent=True)
    graph = build_graph(deps, service.settings)

    with session.lock:
        out = graph.invoke(initial_state(question, session.history))
        answer = Answer.model_validate(out["answer"]) if out["answer"] else None
        # A turn that produced no answer is not history: it would leave a question
        # with no reply in front of the next turn.
        if answer is not None:
            session.history.extend(
                [HumanMessage(content=question), AIMessage(content=answer.text)]
            )

    return AskResponse(
        session_id=chosen,
        answer=answer,
        stop_reason=out["stop_reason"],
        rounds=out["rounds"],
        calls=budget.calls,
        tokens=budget.tokens,
        mode=service.mode,
        latency_s=time.monotonic() - started,
    )


def _parse_boxes(raw: list[str] | None) -> list[BBox]:
    """One ``x0,y0,x1,y1`` per value, normalized to 0..1. A malformed box is a 422."""
    boxes: list[BBox] = []
    for item in raw or []:
        parts = item.split(",")
        if len(parts) != 4:
            raise HTTPException(status_code=422, detail=f"bbox must be x0,y0,x1,y1: {item!r}")
        try:
            x0, y0, x1, y1 = (float(part) for part in parts)
            boxes.append(BBox(x0=x0, y0=y0, x1=x1, y1=y1))
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=f"invalid bbox {item!r}: {exc}") from exc
    return boxes


def _page_png(service: Service, doc_id: str, page: int, raw: list[str] | None) -> Response:
    """The rendered page with the requested boxes drawn. No box means the untouched page."""
    # doc_id becomes a path segment; a separator would read outside the render directory.
    if not doc_id or doc_id in {".", ".."} or "/" in doc_id or "\\" in doc_id:
        raise HTTPException(status_code=404, detail="no such page")
    boxes = _parse_boxes(raw)
    image = page_image(service.settings.render_dir, doc_id, page)
    if image is None:
        raise HTTPException(status_code=404, detail=f"page not rendered: {doc_id}/{page}")
    citations = [Citation(doc_id=doc_id, page=page, bbox=box) for box in boxes]
    canvas = draw_citations(image, citations, doc_id=doc_id, page=page, color=CITED)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


def _default_service() -> Service:
    load_local_env(ENV_PATH)
    return build_service(Settings())


def create_app(factory: Callable[[], Service] | None = None) -> FastAPI:
    """The ASGI app. ``factory`` lets a test supply a service instead of loading a checkpoint."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = (factory or _default_service)()
        yield

    app = FastAPI(title="multimodal-doc-qa", lifespan=lifespan)

    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> AskResponse:
        """One bounded agent loop. An empty page pool is a 200 carrying ``stop_reason``."""
        return answer_question(request.app.state.service, body.question, body.session_id)

    @app.get("/health")
    def health(request: Request) -> dict[str, str | int]:
        service = request.app.state.service
        return {
            "status": "ok",
            "mode": service.mode,
            "artifacts": str(service.artifacts),
            "sessions": len(service.sessions),
        }

    @app.get("/page/{doc_id}/{page}.png")
    def page_png(
        request: Request,
        doc_id: str,
        page: Annotated[int, PathParam(ge=0)],
        bbox: Annotated[list[str] | None, Query()] = None,
    ) -> Response:
        """``?bbox=x0,y0,x1,y1`` repeated once per citation box."""
        return _page_png(request.app.state.service, doc_id, page, bbox)

    return app


app = create_app()


def main() -> None:
    """``python -m multimodal_doc_qa.api``; uvicorn serves the same object by import string."""
    import uvicorn

    settings = Settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
