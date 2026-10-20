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
from pathlib import Path, PurePosixPath
from typing import Annotated

import typer
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi import Path as PathParam
from fastapi.responses import FileResponse, Response
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError

from multimodal_doc_qa.cli import load_deps, remove_document, run_ingest
from multimodal_doc_qa.config import ENV_PATH, Settings, documents_path, load_local_env
from multimodal_doc_qa.graph import build_graph, initial_state
from multimodal_doc_qa.limits import Budget
from multimodal_doc_qa.schemas import Answer, BBox, Citation, load_documents
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
    """What is loaded once at startup. Everything request-scoped is built per request.

    ``retrievers`` holds one built arm per mode, on first use: switching arms is a cache
    lookup, and two users on different arms never block each other. Empty while the
    catalog is empty -- an empty corpus is a valid state rather than a failed start.
    """

    settings: Settings
    mode: str
    artifacts: Path
    sessions: SessionStore
    documents: list[str] = field(default_factory=list)
    retrievers: dict[str, object] = field(default_factory=dict)
    encoder: object = None
    embedder: object = None
    # Serializes ingest + rebuild. A turn keeps answering from the old retriever meanwhile.
    rebuild_lock: threading.Lock = field(default_factory=threading.Lock)
    # One arm may be built while another request is mid-build of a different one.
    retrievers_lock: threading.Lock = field(default_factory=threading.Lock)

    def load_encoders(self) -> None:
        """Both checkpoints ingest needs, built once and reused by every rebuild."""
        if self.encoder is None:
            from multimodal_doc_qa.embed.encoder import MultiVectorEncoder

            self.encoder = MultiVectorEncoder(self.settings.embedder_model, self.settings)
        if self.embedder is None:
            from multimodal_doc_qa.retrievers.text import OcrEmbedder

            self.embedder = OcrEmbedder(self.settings.ocr_embedder_model, self.settings.device)

    def retriever_for(self, mode: str) -> object:
        """The arm for ``mode``, built on first use. ``IndexNotFoundError`` if it cannot be.

        Sharing the encoders matters: building without them would load a second copy of
        a 500M checkpoint for every arm the page is switched to.
        """
        from multimodal_doc_qa.retrievers.assembly import build_retriever

        with self.retrievers_lock:
            if mode not in self.retrievers:
                self.retrievers[mode] = build_retriever(
                    self.settings, mode, encoder=self.encoder, text_embedder=self.embedder
                )
            return self.retrievers[mode]

    def rebuild(self) -> None:
        """Re-read the catalog and drop every arm built from the old index.

        The default arm is rebuilt eagerly so the first question does not pay for it.
        """
        catalog = documents_path(self.artifacts)
        self.documents = list(load_documents(catalog)) if catalog.is_file() else []
        with self.retrievers_lock:
            self.retrievers.clear()
        if not self.documents:
            return
        self.load_encoders()
        self.retriever_for(self.mode)


def available_modes(settings: Settings) -> list[str]:
    """Arms this index can actually run: every index they depend on is on disk.

    The ``abstract`` arms need a VLM pass at ingest, so a plain corpus must not offer them.
    """
    from multimodal_doc_qa.retrievers.assembly import ARMS, COMPONENTS, INDEXES

    return [
        name
        for name, arm in ARMS.items()
        if all(INDEXES[COMPONENTS[stage].index].path(settings).is_file() for stage in arm.stages)
    ]


EMPTY_CORPUS = "语料库还是空的。先在左侧上传 PDF、图片或文本，再提问。"


def build_service(settings: Settings) -> Service:
    """Load whatever the artifacts hold; an empty directory starts without a retriever."""
    service = Service(
        settings=settings,
        mode=settings.mode,
        artifacts=settings.artifacts_dir,
        sessions=SessionStore(settings.session_ttl_seconds, settings.max_sessions),
    )
    service.rebuild()
    return service


class AskRequest(BaseModel):
    """One turn. ``session_id`` continues a conversation; omitting it starts a new one.

    ``mode`` picks the retrieval arm for this turn; omitting it keeps the server default.
    """

    question: str = Field(min_length=1)
    session_id: str | None = None
    mode: str | None = None


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


class IngestSummary(BaseModel):
    """What one upload changed. ``skipped`` are doc_ids already in the catalog."""

    added: list[str]
    skipped: list[str]
    documents: int
    index_mb: float
    elapsed_s: float


class DocumentsResponse(BaseModel):
    """The corpus this server will answer from."""

    mode: str
    documents: list[str]


class DeleteResponse(BaseModel):
    """What is left after a delete. ``documents`` is re-read from the rebuilt catalog."""

    deleted: str
    documents: list[str]


class ModesResponse(BaseModel):
    """The arms this index can run, and the one a request without ``mode`` gets."""

    default: str
    modes: list[str]


def _safe_upload_name(upload: UploadFile) -> str:
    """The basename, so a ``../`` in the client-supplied name cannot write outside the corpus."""
    name = PurePosixPath(upload.filename or "").name
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="upload has no file name")
    return name


def answer_question(
    service: Service, question: str, session_id: str | None = None, mode: str | None = None
) -> AskResponse:
    """Run one turn of a conversation on one retrieval arm.

    The budget is per turn: it is attached to the graph as a callback and spent across
    nodes, so a reused one would charge this ask for the previous ask's calls.

    History is read and extended under the session's lock. Everything else -- the models,
    the graph, the state -- is built per request, so concurrent conversations share
    nothing but the retriever of the arm they asked for.
    """
    from multimodal_doc_qa.retrievers.assembly import MODES, IndexNotFoundError

    started = time.monotonic()
    chosen, session = service.sessions.get_or_create(session_id)
    mode = mode or service.mode
    if mode not in MODES:
        # A client bug, not a state problem: it would have failed against any corpus.
        raise HTTPException(status_code=422, detail=f"unknown mode {mode!r}: try {', '.join(MODES)}")
    if not service.documents:
        # Nothing ingested: running the loop would spend model calls to say so. Not history.
        return AskResponse(
            session_id=chosen,
            answer=Answer(text=EMPTY_CORPUS),
            stop_reason="no_corpus",
            rounds=0,
            calls=0,
            tokens=0,
            mode=mode,
            latency_s=time.monotonic() - started,
        )
    try:
        retriever = service.retriever_for(mode)
    except IndexNotFoundError as exc:
        # Known arm whose index this corpus never got (e.g. the abstract VLM pass).
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    budget = Budget.from_settings(service.settings)
    deps = load_deps(service.settings, retriever, budget, enable_intent=True)
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
        mode=mode,
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


_CORPUS_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt", ".md"}
_APP_HTML = Path(__file__).parent / "ui" / "app.html"


def _ingest_uploads(service: Service, files: list[UploadFile]) -> IngestSummary:
    """Store the uploads in the corpus, then encode the ones not already indexed.

    The corpus directory is the source of truth: ``run_ingest`` re-walks it, so a restart
    can rebuild from the files that are there.
    """
    corpus = service.artifacts / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for upload in files:
        name = _safe_upload_name(upload)
        if Path(name).suffix.lower() not in _CORPUS_SUFFIXES:
            raise HTTPException(status_code=415, detail=f"unsupported file type: {name}")
        data = upload.file.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"{name} is empty")
        (corpus / name).write_bytes(data)

    with service.rebuild_lock:
        service.load_encoders()
        try:
            result = run_ingest(
                corpus,
                service.artifacts,
                append=True,
                vision_encoder=service.encoder,
                ocr_embedder=service.embedder,
            )
        except typer.Exit as exc:
            raise HTTPException(status_code=400, detail="ingest failed; see the server log") from exc
        service.rebuild()

    return IngestSummary(
        added=result.added,
        skipped=result.skipped,
        documents=len(service.documents),
        index_mb=result.index_mb,
        elapsed_s=result.elapsed_s,
    )


def _delete_document(service: Service, doc_id: str) -> DeleteResponse:
    """Remove one document and rebuild. Serialized with ingest by ``rebuild_lock``.

    The rebuild re-encodes the remaining pages, so this holds the lock for as long as an
    upload would; turns in flight keep using the old retriever and finish normally.
    """
    with service.rebuild_lock:
        service.load_encoders()
        try:
            remaining = remove_document(
                doc_id,
                service.artifacts,
                vision_encoder=service.encoder,
                ocr_embedder=service.embedder,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no such document: {doc_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"cannot remove {doc_id}: {exc}") from exc
        except typer.Exit as exc:
            raise HTTPException(status_code=400, detail="rebuild failed; see the server log") from exc
        service.rebuild()
    return DeleteResponse(deleted=doc_id, documents=remaining)


def create_app(factory: Callable[[], Service] | None = None) -> FastAPI:
    """The ASGI app. ``factory`` lets a test supply a service instead of loading a checkpoint."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = (factory or _default_service)()
        yield

    app = FastAPI(title="multimodal-doc-qa", lifespan=lifespan)

    @app.get("/", response_class=FileResponse)
    def index() -> FileResponse:
        """The single-page UI. No build step: one HTML file with plain ``fetch``."""
        return FileResponse(_APP_HTML)

    @app.get("/documents")
    def documents(request: Request) -> DocumentsResponse:
        service = request.app.state.service
        return DocumentsResponse(mode=service.mode, documents=service.documents)

    @app.get("/modes")
    def modes(request: Request) -> ModesResponse:
        service = request.app.state.service
        return ModesResponse(default=service.mode, modes=available_modes(service.settings))

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: str, request: Request) -> DeleteResponse:
        """Delete one document's files and indices, then rebuild from what is left."""
        return _delete_document(request.app.state.service, doc_id)

    @app.post("/ingest")
    def ingest(files: Annotated[list[UploadFile], File()], request: Request) -> IngestSummary:
        """Synchronous on purpose: the browser waits, and there is no job store to poll."""
        return _ingest_uploads(request.app.state.service, files)

    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> AskResponse:
        """One bounded agent loop. An empty page pool is a 200 carrying ``stop_reason``."""
        return answer_question(
            request.app.state.service, body.question, body.session_id, body.mode
        )

    @app.get("/health")
    def health(request: Request) -> dict[str, str | int]:
        service = request.app.state.service
        return {
            "status": "ok",
            "mode": service.mode,
            "artifacts": str(service.artifacts),
            "sessions": len(service.sessions),
            "documents": len(service.documents),
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
