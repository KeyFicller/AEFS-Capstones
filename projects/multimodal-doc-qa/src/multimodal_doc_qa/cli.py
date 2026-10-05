"""CLI: ``ingest`` writes the indices, ``ask`` and ``eval`` read them.

With no subcommand, load the artifacts once and answer questions in a loop.
Heavy imports stay inside the commands so ``--help`` does not load torch.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import typer
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from repl_console.commands import CommandOutcome, LocalCommand

from multimodal_doc_qa.config import (
    ENV_PATH,
    Settings,
    artifact_paths,
    documents_path,
    load_local_env,
    summary_paths,
)
from multimodal_doc_qa.graph import GraphDeps, build_graph, initial_state
from multimodal_doc_qa.limits import Budget
from multimodal_doc_qa.schemas import (
    Answer,
    BBox,
    Citation,
    ImageDocument,
    PdfDocument,
    TextDocument,
    load_documents,
    page_id,
    save_documents,
)
from multimodal_doc_qa.ui import console as ui

app = typer.Typer(help="Agentic RAG over document images", add_completion=False)

_MODE_HELP = (
    "vision: late-interaction over page images. "
    "pool: one vector per page, the mean of those patches. "
    "ocr: page text embedded with the text encoder. "
    "summary: a VLM description of each page, embedded as text. "
    "Set MDQ_MODE to change the default."
)
_MODES = ("vision", "ocr", "pool", "summary")


@dataclass(frozen=True)
class CitedMaterial:
    """One citation resolved to the files and OCR text that back it."""

    doc_id: str
    page: int
    pdf: Path | None
    txt: Path | None
    png: Path
    text: str | None
    bbox: BBox | None


def cited_materials(
    citations: list[Citation],
    artifacts: Path,
    page_texts: dict[tuple[str, int], str],
) -> list[CitedMaterial]:
    """Map each citation to its PDF, rendered page, and OCR text.

    The PDF is the ``<doc_id>.pdf`` sitting in ``artifacts``. Ingest does not copy it there.
    """
    render_dir = artifact_paths(artifacts)[0]
    materials: list[CitedMaterial] = []
    for citation in citations:
        materials.append(
            CitedMaterial(
                doc_id=citation.doc_id,
                page=citation.page,
                pdf=_artifact_file(artifacts, citation.doc_id, ".pdf"),
                txt=_artifact_file(artifacts, citation.doc_id, ".txt"),
                png=render_dir / citation.doc_id / f"p{citation.page:03d}.png",
                text=page_texts.get((citation.doc_id, citation.page)),
                bbox=citation.bbox,
            )
        )
    return materials


def _artifact_file(artifacts: Path, doc_id: str, suffix: str) -> Path | None:
    """``<doc_id><suffix>``, or the file whose name is already ``doc_id``.

    A shared stem makes the id the filename (``Test.pdf``). That file is not ``Test.pdf.pdf``.
    """
    candidate = artifacts / f"{doc_id}{suffix}"
    if candidate.is_file():
        return candidate
    named = artifacts / doc_id
    if named.is_file() and named.suffix.lower() == suffix:
        return named
    return None


def _ocr_page_texts(settings: Settings) -> dict[tuple[str, int], str]:
    """Page text from the OCR index, for display. Missing index means no text lines."""
    import torch

    _, _, ocr_path = artifact_paths(settings.artifacts_dir)
    if not ocr_path.is_file():
        return {}
    grouped: dict[tuple[str, int], list[str]] = {}
    for doc_id, payload in torch.load(ocr_path, weights_only=True).items():
        for text, page in payload["chunks"]:
            grouped.setdefault((doc_id, int(page)), []).append(text)
    return {key: "\n".join(parts) for key, parts in grouped.items()}


def _render_turn(
    answer: Answer | None,
    out: dict,
    budget: Budget,
    settings: Settings,
    artifacts: Path,
    page_texts: dict[tuple[str, int], str],
    *,
    question: str,
    mode: str,
) -> None:
    from multimodal_doc_qa.ui.viewer import append_turn

    if answer is None:
        ui.render_error(f"no answer (stop_reason={out['stop_reason']})")
    else:
        ui.render_reply(answer.text)
        ui.render_sources(cited_materials(answer.citations, artifacts, page_texts))
    append_turn(
        artifacts / "turns.jsonl",
        mode=mode,
        question=question,
        answer=answer,
        rounds=out["rounds"],
        calls=budget.calls,
        tokens=budget.tokens,
        stop_reason=out["stop_reason"],
    )
    ui.render_budget(
        rounds=out["rounds"],
        max_rounds=settings.max_rounds,
        calls=budget.calls,
        max_calls=settings.max_ask_calls,
        tokens=budget.tokens,
        max_tokens=settings.max_ask_tokens,
        elapsed=budget.elapsed(),
        max_seconds=settings.max_ask_seconds,
        stop_reason=out["stop_reason"],
    )


_INGEST_CORPUS = typer.Option(
    ..., "--corpus", help="Directory of PDF, image, .txt, or .md documents"
)
_INGEST_OUT = typer.Option(
    None, "--out", help="Artifacts directory (default: settings.artifacts_dir)"
)


@app.command()
def ingest(
    corpus: Path = _INGEST_CORPUS,
    out: Path | None = _INGEST_OUT,
) -> None:
    """Persist both indices. One ``ask`` uses one; the ablation needs both.

    A PDF or a standalone image is encoded with ``encode_images``. Plain text is
    chunked and encoded with ``encode_texts``. Nothing rasterizes a text file.
    """
    import torch

    from multimodal_doc_qa.embed.encoder import MultiVectorEncoder
    from multimodal_doc_qa.index.maxsim import MultiVectorIndex
    from multimodal_doc_qa.retrievers.text import OcrEmbedder

    settings = Settings()
    artifacts = (out or settings.artifacts_dir).expanduser()
    render_dir, vision_path, ocr_path = artifact_paths(artifacts)
    sources = _unique_files(_checked_sources(corpus.expanduser()))

    artifacts.mkdir(parents=True, exist_ok=True)
    vision = MultiVectorIndex(device="cpu")
    vision_encoder = MultiVectorEncoder(settings.embedder_model, settings)
    ocr_embedder = OcrEmbedder(settings.ocr_embedder_model, settings.device)
    ui.echo(
        f"vision encoder  {type(vision_encoder.model).__name__} {settings.embedder_model} "
        f"({settings.device}/{settings.dtype})"
    )
    ui.echo(f"ocr encoder     {settings.ocr_embedder_model} ({settings.device})")

    documents: list[ImageDocument | PdfDocument | TextDocument] = []
    ocr_docs: dict[str, dict] = {}
    for source, doc_id in _assign_doc_ids(sources):
        document, payload = _index_source(
            source, doc_id, artifacts, render_dir, vision, vision_encoder, ocr_embedder
        )
        documents.append(document)
        ocr_docs[document.doc_id] = payload

    vision.save(vision_path)
    ocr_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ocr_docs, ocr_path)
    save_documents(documents, documents_path(artifacts))
    if settings.mode == "summary" or settings.summaries:
        _write_summaries(settings, artifacts, render_dir, documents, ocr_embedder.encode)

    ui.echo(f"vision    {vision_path}  ({vision.nbytes() / 1e6:.1f} MB)")
    ui.echo(f"ocr       {ocr_path}")
    ui.echo(f"[green]ingested[/] {len(sources)} docs -> {artifacts}")


def _write_summaries(settings, artifacts, render_dir, documents, encode) -> None:
    """Bind one description to each page. The cache means a repeated ingest does not call the VLM again."""
    from multimodal_doc_qa.summarize import index_summaries, load_cache

    cache_path, summary_path = summary_paths(artifacts)
    cache = load_cache(cache_path)
    payload = index_summaries(
        documents, render_dir, encode, cache, cache_path, _page_describer(settings)
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(payload, summary_path)
    ui.echo(f"summary   {summary_path}  ({len(cache)} cached pages)")


def _page_describer(settings: Settings):
    """A ``png -> text`` callable. The chat model is loaded on the first uncached page."""
    model = None

    def describe(png: Path) -> str:
        nonlocal model
        if model is None:
            from multimodal_doc_qa.synth.answer import build_chat_model

            model = build_chat_model(settings)
        from multimodal_doc_qa.summarize import describe_page

        return describe_page(model, png)

    return describe


def _checked_sources(corpus: Path) -> list[Path]:
    """Every document in ``corpus``. A missing or empty directory stops ingest."""
    if not corpus.is_dir():
        ui.render_error(f"no such corpus directory: {corpus}")
        raise typer.Exit(code=1)
    sources = _corpus_sources(corpus)
    if not sources:
        ui.render_error(f"no documents in {corpus}")
        raise typer.Exit(code=1)
    return sources


def _assign_doc_ids(sources: list[Path]) -> list[tuple[Path, str]]:
    """Use the stem. A stem shared by two files becomes the filename, so both can be stored."""
    stems = [source.stem for source in sources]
    shared = {stem for stem in stems if stems.count(stem) > 1}
    return [(source, source.name if source.stem in shared else source.stem) for source in sources]


def _unique_files(sources: list[Path]) -> list[Path]:
    """Drop a later file whose bytes match an earlier one. The kept name is the first in sort order."""
    seen: dict[str, Path] = {}
    kept: list[Path] = []
    for source in sources:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        previous = seen.get(digest)
        if previous is not None:
            ui.echo(f"  skip {source.name}: same bytes as {previous.name}")
            continue
        seen[digest] = source
        kept.append(source)
    return kept


def _index_source(source, doc_id, artifacts, render_dir, vision, vision_encoder, ocr_embedder):
    """Encode one file into the vision index and return its OCR payload."""
    from multimodal_doc_qa.baseline.ocr import chunk_texts

    document, page_texts = _ingest_source(source, doc_id, render_dir, vision, vision_encoder)
    chunks = chunk_texts(page_texts) or [("", 0)]
    # A shared stem makes doc_id the filename. Appending .txt would write Test.pdf.txt.
    if not Path(document.doc_id).suffix:
        sidecar = artifacts / f"{document.doc_id}.txt"
        if sidecar.resolve() != source.resolve():
            sidecar.write_text("\n\n".join(page_texts), encoding="utf-8")
    ui.echo(
        f"  {document.doc_id}: {document.origin} {len(page_texts)} pages, {len(chunks)} text chunks"
    )
    return document, {"index": ocr_embedder.encode([text for text, _ in chunks]), "chunks": chunks}


_TEXT_SUFFIXES = {".txt", ".md"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_CORPUS_SUFFIXES = {".pdf", *_TEXT_SUFFIXES, *_IMAGE_SUFFIXES}


def _ingest_source(source, doc_id, render_dir, vision, vision_encoder):
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return _ingest_pdf(source, doc_id, render_dir, vision, vision_encoder)
    if suffix in _TEXT_SUFFIXES:
        return _ingest_text(source, doc_id, vision, vision_encoder)
    return _ingest_image(source, doc_id, render_dir, vision, vision_encoder)


def _ingest_text(source, doc_id, vision, vision_encoder):
    pages = _text_pages(source.read_text(encoding="utf-8"))
    if not pages:
        ui.render_error(f"{source.name} is empty")
        raise typer.Exit(code=1)
    for index, vectors in enumerate(vision_encoder.encode_texts(pages)):
        vision.add(page_id(doc_id, index), vectors)
    return TextDocument(doc_id=doc_id, pages=pages), pages


def _ingest_pdf(source, doc_id, render_dir, vision, vision_encoder):
    from PIL import Image

    from multimodal_doc_qa.baseline.ocr import extract_page_texts
    from multimodal_doc_qa.render.renderer import render_pages

    pages = render_pages(source, render_dir / doc_id)
    for page in pages:
        with Image.open(page) as image:
            vectors = vision_encoder.encode_images([image.convert("RGB")])[0]
        vision.add(page_id(doc_id, int(page.stem[1:])), vectors)
    return PdfDocument(doc_id=doc_id), extract_page_texts(source, pages)


def _ingest_image(source, doc_id, render_dir, vision, vision_encoder):
    from PIL import Image

    from multimodal_doc_qa.baseline.ocr import extract_image_text

    dest = render_dir / doc_id / "p000.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        rgb.save(dest)
        vectors = vision_encoder.encode_images([rgb])[0]
        text = extract_image_text(rgb)
    vision.add(page_id(doc_id, 0), vectors)
    return ImageDocument(doc_id=doc_id), [text]


def _text_pages(raw: str) -> list[str]:
    """One retrieval page per chunk. The chunk index is the page number."""
    from multimodal_doc_qa.baseline.ocr import chunk_texts

    return [text for text, _ in chunk_texts([raw])]


def _corpus_sources(corpus: Path) -> list[Path]:
    """PDF, image, and plain-text documents in ``corpus``. Names sort so ingest order is stable."""
    return sorted(
        path
        for path in corpus.iterdir()
        if path.is_file() and path.suffix.lower() in _CORPUS_SUFFIXES
    )


class IndexNotFoundError(Exception):
    """The selected retrieval path has no index on disk."""


def _resolve_mode(mode: str | None, settings: Settings) -> str:
    """CLI ``--mode`` wins. Otherwise ``MDQ_MODE`` (default ``vision``)."""
    chosen = mode or settings.mode
    if chosen not in _MODES:
        raise typer.BadParameter(f"mode must be one of {', '.join(_MODES)}, got {chosen!r}")
    return chosen


def _load_retriever(settings: Settings, mode: str, *, encoder: object = None, text_embedder: object = None):
    """Build the retriever for ``mode``. Pass an encoder to share it across arms."""
    _, vision_path, ocr_path = artifact_paths(settings.artifacts_dir)
    if mode == "vision":
        return _load_vision(settings, vision_path, encoder)
    if mode == "pool":
        return _load_pool(settings, vision_path, encoder)
    if mode == "ocr":
        return _load_text_index(settings, ocr_path, "OCR", text_embedder)
    if mode == "summary":
        return _load_text_index(
            settings, summary_paths(settings.artifacts_dir)[1], "summary", text_embedder
        )
    raise typer.BadParameter(f"mode must be one of {', '.join(_MODES)}, got {mode!r}")


def _load_vision(settings: Settings, vision_path: Path, encoder: object = None):
    from multimodal_doc_qa.embed.encoder import MultiVectorEncoder
    from multimodal_doc_qa.index.maxsim import MultiVectorIndex
    from multimodal_doc_qa.retrievers.multivector import MultiVectorRetriever

    if not vision_path.is_file():
        raise IndexNotFoundError(f"no vision index at {vision_path} -- run `doc-qa ingest` first")
    return MultiVectorRetriever(
        encoder=encoder or MultiVectorEncoder(settings.embedder_model, settings),
        index=MultiVectorIndex.load(vision_path),
        k=settings.top_k,
        min_score_ratio=settings.min_score_ratio,
    )


def _load_pool(settings: Settings, vision_path: Path, encoder: object = None):
    from multimodal_doc_qa.embed.encoder import MultiVectorEncoder
    from multimodal_doc_qa.index.maxsim import MultiVectorIndex
    from multimodal_doc_qa.retrievers.pool import PooledRetriever, mean_vector

    if not vision_path.is_file():
        raise IndexNotFoundError(f"no vision index at {vision_path} -- run `doc-qa ingest` first")
    index = MultiVectorIndex.load(vision_path)
    return PooledRetriever(
        encoder=encoder or MultiVectorEncoder(settings.embedder_model, settings),
        pages={page_id: mean_vector(matrix) for page_id, matrix in index.matrices().items()},
        k=settings.top_k,
        min_score_ratio=settings.min_score_ratio,
    )


def _load_text_index(settings: Settings, path: Path, label: str, embedder: object = None):
    import torch

    from multimodal_doc_qa.retrievers.text import MultiDocTextRetriever, OcrEmbedder, TextRetriever

    if not path.is_file():
        hint = " -- run `doc-qa ingest` first"
        if label == "summary":
            hint = " -- re-run ingest with MDQ_SUMMARIES=1 or MDQ_MODE=summary"
        raise IndexNotFoundError(f"no {label} index at {path}{hint}")
    if embedder is None:
        embedder = OcrEmbedder(settings.ocr_embedder_model, settings.device)
    return MultiDocTextRetriever(
        retrievers=[
            TextRetriever(
                doc_id=doc_id,
                embedder=embedder,
                index=payload["index"],
                chunks=payload["chunks"],
                k=settings.top_k,
            )
            for doc_id, payload in torch.load(path, weights_only=True).items()
        ],
        k=settings.top_k,
        min_score_ratio=settings.min_score_ratio,
    )


def _open_retriever(settings: Settings, mode: str):
    """Load one retrieval path, or stop the process when its index is missing."""
    try:
        return _load_retriever(settings, mode)
    except IndexNotFoundError as exc:
        ui.render_error(str(exc))
        raise typer.Exit(code=1) from exc


def _load_deps(settings: Settings, retriever: object, budget: Budget) -> GraphDeps:
    """Wire the graph's models. One chat model serves the three nodes and the answerer."""
    from multimodal_doc_qa.agent.nodes import Followups, Subqueries, Unsupported
    from multimodal_doc_qa.synth.answer import AnswerSynthesizer, build_chat_model

    chat = build_chat_model(settings)
    catalog = documents_path(settings.artifacts_dir)
    documents = load_documents(catalog) if catalog.is_file() else None
    render_dir = artifact_paths(settings.artifacts_dir)[0]
    rerank = None
    if settings.rerank:
        from multimodal_doc_qa.retrievers.rerank import PageRank, rerank_pages

        ranker = chat.with_structured_output(PageRank)

        def rerank(query, docs, _ranker=ranker, _render=render_dir, _docs=documents):
            return rerank_pages(_ranker, query, docs, _render, _docs)

    return GraphDeps(
        retriever=retriever,
        planner_model=chat.with_structured_output(Subqueries),
        assessor_model=chat.with_structured_output(Followups),
        verifier_model=chat.with_structured_output(Unsupported),
        synth=AnswerSynthesizer(settings),
        render_dir=render_dir,
        documents=documents,
        budget=budget,
        rerank=rerank,
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from the ingested corpus"),
    mode: str | None = typer.Option(None, "--mode", help=_MODE_HELP),
) -> None:
    """Answer one question through the agentic graph."""
    settings = Settings()
    mode = _resolve_mode(mode, settings)

    retriever = _open_retriever(settings, mode)
    budget = Budget.from_settings(settings)
    deps = _load_deps(settings, retriever, budget)

    with ui.working():
        out = build_graph(deps, settings).invoke(initial_state(question))
    answer = Answer.model_validate(out["answer"]) if out["answer"] else None
    _render_turn(
        answer,
        out,
        budget,
        settings,
        settings.artifacts_dir,
        _ocr_page_texts(settings),
        question=question,
        mode=mode,
    )
    if answer is None:
        raise typer.Exit(code=1)


_EVAL_QUESTIONS = typer.Option(
    ..., "--questions", help="Questions JSON: gold answer and evidence"
)
_EVAL_MODE = typer.Option(None, "--mode", help=_MODE_HELP)
_EVAL_OUT = typer.Option(None, "--out", help="results.jsonl to append to")


@app.command("eval")
def eval_questions(
    questions: Path = _EVAL_QUESTIONS,
    mode: str | None = _EVAL_MODE,
    out: Path | None = _EVAL_OUT,
    iou_threshold: float = typer.Option(
        0.5, "--iou-threshold", help="IoU above which a citation matches a gold box"
    ),
    max_seconds: float | None = typer.Option(
        None, "--max-seconds", help="Stop the suite after this long and keep the rows so far"
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Stop the suite after this many tokens and keep the rows so far"
    ),
) -> None:
    """One graph per question. A reused graph would carry the previous ask's budget."""
    from multimodal_doc_qa.eval.run import RESULTS_PATH, run_eval

    settings = Settings()
    mode = _resolve_mode(mode, settings)
    items = _load_questions(questions)
    retriever = _open_retriever(settings, mode)
    written = out or RESULTS_PATH
    summary = run_eval(
        items,
        lambda question: _eval_one(question, settings, retriever),
        out_path=written,
        settings=settings,
        mode=mode,
        k=settings.top_k,
        iou_threshold=iou_threshold,
        max_tokens=max_tokens,
        max_seconds=max_seconds,
    )
    _print_eval(summary, settings, mode, iou_threshold, written)


def _load_questions(questions: Path):
    import json

    from multimodal_doc_qa.schemas import Question

    if not questions.is_file():
        ui.render_error(f"no questions at {questions}")
        raise typer.Exit(code=1)
    return [Question.model_validate(item) for item in json.loads(questions.read_text())]


def _eval_one(question, settings: Settings, retriever):
    from multimodal_doc_qa.eval.run import QuestionRun

    budget = Budget.from_settings(settings)
    deps = _load_deps(settings, retriever, budget)
    with ui.working():
        out_state = build_graph(deps, settings).invoke(initial_state(question.text))
    return QuestionRun(
        answer=Answer.model_validate(out_state["answer"]) if out_state["answer"] else None,
        ranked=[doc.metadata["page_id"] for doc in retriever.invoke(question.text)],
        pool=out_state["page_ids"],
        rounds=out_state["rounds"],
        calls=budget.calls,
        tokens=budget.tokens,
        stop_reason=out_state["stop_reason"],
    )


def _print_eval(
    summary: dict, settings: Settings, mode: str, iou_threshold: float, written: Path
) -> None:
    ui.echo(f"[bold]{mode}[/] over {summary['n_done']}/{summary['n_questions']} questions")
    ui.echo(
        f"nDCG@{settings.top_k} {summary['ndcg_at_k']:.4f}"
        f"  IoU@{iou_threshold:g} {summary['iou_at_threshold']:.4f}"
        f"  (strict containment {summary['bbox_hit_rate']:.4f})"
    )
    ui.echo(
        f"latency p50/p95 {summary['latency_p50_s']:.2f}s/{summary['latency_p95_s']:.2f}s"
        f"  tokens {summary['tokens']}  stop_reasons {summary['stop_reasons']}"
    )
    if summary["stopped_after"]:
        ui.echo(f"[yellow]cut short before {summary['stopped_after']}[/]")
    ui.echo(f"written  {written}")


def _mode_models(settings: Settings, mode: str) -> str:
    return ui.mode_models(
        mode,
        vision=settings.embedder_model,
        ocr=settings.ocr_embedder_model,
        describer=settings.answerer_model,
    )


def _preload_retrievers(settings: Settings) -> tuple[dict[str, object], dict[str, str]]:
    """Load each encoder once, then every arm whose index is on disk.

    Shift-Tab only swaps the already-built retriever. A missing index is recorded
    and skipped; the models stay loaded.
    """
    from multimodal_doc_qa.embed.encoder import MultiVectorEncoder
    from multimodal_doc_qa.retrievers.text import OcrEmbedder

    encoder = MultiVectorEncoder(settings.embedder_model, settings)
    text_embedder = OcrEmbedder(settings.ocr_embedder_model, settings.device)
    loaded: dict[str, object] = {}
    missing: dict[str, str] = {}
    for name in _MODES:
        try:
            loaded[name] = _load_retriever(
                settings, name, encoder=encoder, text_embedder=text_embedder
            )
        except IndexNotFoundError as exc:
            missing[name] = str(exc)
    return loaded, missing


def _corpus_under_artifacts(artifacts: Path, arg: str) -> Path | None:
    """Resolve `arg` under `artifacts`. `.` is `artifacts` itself."""
    raw = Path(arg).expanduser()
    root = artifacts.expanduser().resolve()
    candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if candidate != root and root not in candidate.parents:
        return None
    return candidate


class IndexCommand(LocalCommand):
    """Rebuild the artifacts index from a corpus directory and use it immediately."""

    name = "index"
    summary = "rebuild the index from a corpus under artifacts"

    def __init__(self, settings, artifacts, loaded, missing, page_texts, history, active) -> None:
        self._settings = settings
        self._artifacts = artifacts
        self._loaded = loaded
        self._missing = missing
        self._page_texts = page_texts
        self._history = history
        self._active = active

    def run(self, ctx, args: str) -> CommandOutcome:
        text = args.strip()
        if not text:
            return CommandOutcome(message="usage: /index <corpus-dir>")
        corpus = _corpus_under_artifacts(self._artifacts, text)
        if corpus is None:
            return CommandOutcome(message="path escapes artifacts")
        try:
            ingest(corpus=corpus, out=None)
        except typer.Exit:
            return CommandOutcome()
        new_loaded, new_missing = _preload_retrievers(self._settings)
        if not new_loaded:
            mode = self._active["mode"]
            ui.render_error(new_missing.get(mode) or next(iter(new_missing.values())))
            return CommandOutcome()
        self._loaded.clear()
        self._loaded.update(new_loaded)
        self._missing.clear()
        self._missing.update(new_missing)
        self._page_texts.clear()
        self._page_texts.update(_ocr_page_texts(self._settings))
        self._history.clear()
        mode = self._active["mode"]
        if mode not in self._loaded:
            mode = next(name for name in _MODES if name in self._loaded)
            self._active["mode"] = mode
            return CommandOutcome(message=f"using the new index ({mode})")
        return CommandOutcome(message="using the new index")


def _repl(mode: str) -> None:
    """Load every retrieval arm up front, then answer questions until the user stops.

    A blank line is ignored. ``/quit`` exits. Ctrl-C during a turn cancels that turn only.
    Shift-Tab switches the retrieval path; the status bar names that path's models.
    Each question compiles its own graph. Earlier turns are messages, not a joined string.
    """
    from repl_console import Repl

    settings = Settings()
    artifacts = settings.artifacts_dir
    loaded, missing = _preload_retrievers(settings)
    if mode not in loaded:
        ui.render_error(missing[mode])
        raise typer.Exit(code=1)
    page_texts = _ocr_page_texts(settings)
    active = {"mode": mode}
    history: list[BaseMessage] = []

    try:
        from prompt_toolkit.key_binding import KeyBindings
    except ImportError:  # pragma: no cover - the dev env has prompt_toolkit
        bindings = None
    else:
        def switch(event) -> None:
            nxt = ui.other_mode(active["mode"])
            if nxt not in loaded:
                ui.render_error(missing[nxt])
                return
            active["mode"] = nxt
            event.app.invalidate()

        bindings = KeyBindings()
        bindings.add("s-tab")(switch)

    def on_task(message) -> None:
        question = str(message.content)
        _run_turn(
            settings,
            loaded[active["mode"]],
            question,
            history,
            artifacts,
            page_texts,
            active["mode"],
        )

    Repl(
        title="multimodal-doc-qa",
        info={
            "artifacts": str(artifacts),
            "mode": mode,
            "models": _mode_models(settings, mode),
        },
        root=Path.cwd(),
        extra_dirs=(artifacts,),
        commands=(
            IndexCommand(settings, artifacts, loaded, missing, page_texts, history, active),
        ),
        on_task=on_task,
        toolbar=lambda: ui.status_bar(active["mode"], _mode_models(settings, active["mode"])),
        key_bindings=bindings,
    ).run()


def _run_turn(
    settings, retriever, question: str, history: list[BaseMessage], artifacts, page_texts, mode: str
) -> None:
    """One ask. A failure is printed and the session stays up."""
    try:
        budget = Budget.from_settings(settings)
        deps = _load_deps(settings, retriever, budget)
        with ui.working():
            out = build_graph(deps, settings).invoke(initial_state(question, history))
    except KeyboardInterrupt:
        ui.render_error("turn cancelled")
        return
    except Exception as exc:  # noqa: BLE001 - one bad turn must not end the session
        ui.render_error(str(exc))
        return

    answer = Answer.model_validate(out["answer"]) if out["answer"] else None
    _render_turn(
        answer, out, budget, settings, artifacts, page_texts, question=question, mode=mode
    )
    if answer is not None:
        history.extend([HumanMessage(content=question), AIMessage(content=answer.text)])


@app.command()
def chat(
    mode: str | None = typer.Option(None, "--mode", help=_MODE_HELP),
) -> None:
    """Load the artifacts once and answer questions until /quit, Ctrl-C, or Ctrl-D."""
    settings = Settings()
    _repl(_resolve_mode(mode, settings))


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    mode: str | None = typer.Option(None, "--mode", help=_MODE_HELP),
) -> None:
    """Load ``local.env``, then start the REPL when no subcommand is given.

    ``--mode`` selects vision, pool, ocr, or summary. Omit it to use ``MDQ_MODE``.
    """
    load_local_env(ENV_PATH)
    if ctx.invoked_subcommand is None:
        _repl(_resolve_mode(mode, Settings()))


if __name__ == "__main__":
    app()
