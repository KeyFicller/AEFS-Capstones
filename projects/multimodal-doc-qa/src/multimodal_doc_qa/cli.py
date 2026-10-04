"""CLI: ``ingest`` writes the indices, ``ask`` and ``eval`` read them.

With no subcommand, load the artifacts once and answer questions in a loop.
Heavy imports stay inside the commands so ``--help`` does not load torch.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import typer
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from multimodal_doc_qa.budget import Budget
from multimodal_doc_qa.config import (
    ENV_PATH,
    Settings,
    artifact_paths,
    documents_path,
    load_local_env,
)
from multimodal_doc_qa.graph import GraphDeps, build_graph, initial_state
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
    "vision: encode page images and retrieve by late interaction. "
    "ocr: extract page text, embed it, and retrieve by similarity."
)


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
) -> None:
    if answer is None:
        ui.render_error(f"no answer (stop_reason={out['stop_reason']})")
    else:
        ui.render_reply(answer.text)
        ui.render_sources(cited_materials(answer.citations, artifacts, page_texts))
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


@app.command()
def ingest(
    corpus: Path = typer.Option(
        ..., "--corpus", help="Directory of PDF, image, .txt, or .md documents"
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Artifacts directory (default: settings.artifacts_dir)"
    ),
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

    ui.echo(f"vision    {vision_path}  ({vision.nbytes() / 1e6:.1f} MB)")
    ui.echo(f"ocr       {ocr_path}")
    ui.echo(f"[green]ingested[/] {len(sources)} docs -> {artifacts}")


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


def _load_retriever(settings: Settings, mode: str):
    """Build the vision or OCR retriever. Raises ``BadParameter`` for any other mode."""
    _, vision_path, ocr_path = artifact_paths(settings.artifacts_dir)
    if mode == "vision":
        return _load_vision(settings, vision_path)
    if mode == "ocr":
        return _load_ocr(settings, ocr_path)
    raise typer.BadParameter(f"mode must be vision or ocr, got {mode!r}")


def _load_vision(settings: Settings, vision_path: Path):
    from multimodal_doc_qa.embed.encoder import MultiVectorEncoder
    from multimodal_doc_qa.index.maxsim import MultiVectorIndex
    from multimodal_doc_qa.retrievers.multivector import MultiVectorRetriever

    if not vision_path.is_file():
        raise IndexNotFoundError(f"no vision index at {vision_path} -- run `doc-qa ingest` first")
    return MultiVectorRetriever(
        encoder=MultiVectorEncoder(settings.embedder_model, settings),
        index=MultiVectorIndex.load(vision_path),
        k=settings.top_k,
        min_score_ratio=settings.min_score_ratio,
    )


def _load_ocr(settings: Settings, ocr_path: Path):
    import torch

    from multimodal_doc_qa.retrievers.text import MultiDocTextRetriever, OcrEmbedder, TextRetriever

    if not ocr_path.is_file():
        raise IndexNotFoundError(f"no OCR index at {ocr_path} -- run `doc-qa ingest` first")
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
            for doc_id, payload in torch.load(ocr_path, weights_only=True).items()
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
    return GraphDeps(
        retriever=retriever,
        planner_model=chat.with_structured_output(Subqueries),
        assessor_model=chat.with_structured_output(Followups),
        verifier_model=chat.with_structured_output(Unsupported),
        synth=AnswerSynthesizer(settings),
        render_dir=artifact_paths(settings.artifacts_dir)[0],
        documents=load_documents(catalog) if catalog.is_file() else None,
        budget=budget,
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from the ingested corpus"),
    mode: str = typer.Option("vision", "--mode", help=_MODE_HELP),
) -> None:
    """Answer one question through the agentic graph."""
    settings = Settings()

    retriever = _open_retriever(settings, mode)
    budget = Budget.from_settings(settings)
    deps = _load_deps(settings, retriever, budget)

    with ui.working():
        out = build_graph(deps, settings).invoke(initial_state(question))
    answer = Answer.model_validate(out["answer"]) if out["answer"] else None
    _render_turn(answer, out, budget, settings, settings.artifacts_dir, _ocr_page_texts(settings))
    if answer is None:
        raise typer.Exit(code=1)


@app.command("eval")
def eval_questions(
    questions: Path | None = typer.Option(
        None, "--questions", help="Holdout JSON (default: the corpus artifacts)"
    ),
    mode: str = typer.Option("vision", "--mode", help=_MODE_HELP),
    out: Path | None = typer.Option(None, "--out", help="results.jsonl to append to"),
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


def _load_questions(questions: Path | None):
    import json

    from multimodal_doc_qa.schemas import Question

    source = questions or (
        Path(__file__).resolve().parent / "corpus" / "_artifacts" / "questions.json"
    )
    if not source.is_file():
        ui.render_error(f"no questions at {source}")
        raise typer.Exit(code=1)
    return [Question.model_validate(item) for item in json.loads(source.read_text())]


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


def _embedder_name(settings: Settings, mode: str) -> str:
    return settings.embedder_model if mode == "vision" else settings.ocr_embedder_model


def _repl(mode: str) -> None:
    """Load both retrieval paths, then answer questions until the user stops.

    A blank line is ignored. ``:q`` exits. Ctrl-C during a turn cancels that turn only.
    Shift-Tab switches the retrieval path; the status bar shows the one in force.
    Each question compiles its own graph. Earlier turns are messages, not a joined string.
    """
    settings = Settings()
    artifacts = settings.artifacts_dir
    loaded = {path: _open_retriever(settings, path) for path in (mode, ui.other_mode(mode))}
    page_texts = _ocr_page_texts(settings)
    ui.banner(
        artifacts=str(artifacts),
        mode=mode,
        embedder=_embedder_name(settings, mode),
        answerer=settings.answerer_model,
    )

    active = {"mode": mode}

    def switch() -> None:
        active["mode"] = ui.other_mode(active["mode"])

    prompt = ui.Prompt(
        on_switch=switch,
        status=lambda: ui.status_bar(active["mode"], _embedder_name(settings, active["mode"])),
    )
    history: list[BaseMessage] = []
    while True:
        question = _read_question(prompt)
        if question is None:
            return
        if not question:
            continue
        _run_turn(settings, loaded[active["mode"]], question, history, artifacts, page_texts)


def _read_question(prompt: ui.Prompt) -> str | None:
    """The next line. ``None`` leaves the REPL; an empty string is ignored."""
    try:
        question = prompt.ask().strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if question in {":q", "quit", "exit"}:
        return None
    return question


def _run_turn(
    settings, retriever, question: str, history: list[BaseMessage], artifacts, page_texts
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
    _render_turn(answer, out, budget, settings, artifacts, page_texts)
    if answer is not None:
        history.extend([HumanMessage(content=question), AIMessage(content=answer.text)])


@app.command()
def chat(
    mode: str = typer.Option("vision", "--mode", help=_MODE_HELP),
) -> None:
    """Load the artifacts once and answer questions until :q, Ctrl-C, or Ctrl-D."""
    _repl(mode)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    mode: str = typer.Option("vision", "--mode", help=_MODE_HELP),
) -> None:
    """Load ``local.env``, then start the REPL when no subcommand is given.

    ``--mode vision`` retrieves encoded page images. ``--mode ocr`` retrieves
    embedded page text. A subcommand uses its own ``--mode``.
    """
    load_local_env(ENV_PATH)
    if ctx.invoked_subcommand is None:
        _repl(mode)


if __name__ == "__main__":
    app()
