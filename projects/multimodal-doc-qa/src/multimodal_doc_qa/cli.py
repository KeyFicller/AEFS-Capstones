"""CLI: ``ingest`` writes the indices, ``ask`` and ``eval`` read them.

With no subcommand, load the artifacts once and answer questions in a loop.
Heavy imports stay inside the commands so ``--help`` does not load torch.
"""

from dataclasses import dataclass
from pathlib import Path

import typer
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from multimodal_doc_qa.budget import Budget
from multimodal_doc_qa.config import ENV_PATH, Settings, artifact_paths, load_local_env
from multimodal_doc_qa.graph import GraphDeps, build_graph, initial_state
from multimodal_doc_qa.schemas import Answer, BBox, Citation, page_id
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
        pdf = artifacts / f"{citation.doc_id}.pdf"
        materials.append(
            CitedMaterial(
                doc_id=citation.doc_id,
                page=citation.page,
                pdf=pdf if pdf.is_file() else None,
                png=render_dir / citation.doc_id / f"p{citation.page:03d}.png",
                text=page_texts.get((citation.doc_id, citation.page)),
                bbox=citation.bbox,
            )
        )
    return materials


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
        ..., "--corpus", help="Directory holding one <doc_id>.pdf per document"
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Artifacts directory (default: settings.artifacts_dir)"
    ),
) -> None:
    """Render every PDF and persist both indices. One ``ask`` uses one; the ablation needs both."""
    import torch
    from PIL import Image

    from multimodal_doc_qa.baseline.ocr import chunk_texts, extract_page_texts
    from multimodal_doc_qa.embed.encoder import MultiVectorEncoder
    from multimodal_doc_qa.index.maxsim import MultiVectorIndex
    from multimodal_doc_qa.render.renderer import render_pages
    from multimodal_doc_qa.retrievers.text import OcrEmbedder

    settings = Settings()
    artifacts = (out or settings.artifacts_dir).expanduser()
    render_dir, vision_path, ocr_path = artifact_paths(artifacts)

    pdfs = sorted(corpus.expanduser().glob("*.pdf"))
    if not pdfs:
        ui.render_error(f"no PDFs in {corpus}")
        raise typer.Exit(code=1)

    vision = MultiVectorIndex(device="cpu")
    ocr_docs: dict[str, dict] = {}

    vision_encoder = MultiVectorEncoder(settings.embedder_model, settings)
    ocr_embedder = OcrEmbedder(settings.ocr_embedder_model, settings.device)
    ui.echo(
        f"vision encoder  {type(vision_encoder.model).__name__} {settings.embedder_model} "
        f"({settings.device}/{settings.dtype})"
    )
    ui.echo(f"ocr encoder     {settings.ocr_embedder_model} ({settings.device})")

    for pdf in pdfs:
        doc_id = pdf.stem
        pages = render_pages(pdf, render_dir / doc_id)

        for page in pages:
            with Image.open(page) as image:
                vectors = vision_encoder.encode_images([image.convert("RGB")])[0]
            vision.add(page_id(doc_id, int(page.stem[1:])), vectors)

        chunks = chunk_texts(extract_page_texts(pdf, pages))
        ocr_docs[doc_id] = {
            "index": ocr_embedder.encode([text for text, _ in chunks]),
            "chunks": chunks,
        }
        ui.echo(f"  {doc_id}: {len(pages)} pages, {len(chunks)} text chunks")

    vision.save(vision_path)
    ocr_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ocr_docs, ocr_path)

    ui.echo(f"vision    {vision_path}  ({vision.nbytes() / 1e6:.1f} MB)")
    ui.echo(f"ocr       {ocr_path}")
    ui.echo(f"[green]ingested[/] {len(pdfs)} docs -> {artifacts}")


class IndexNotFoundError(Exception):
    """The selected retrieval path has no index on disk."""


def _load_retriever(settings: Settings, mode: str):
    """Build the vision or OCR retriever. Raises ``BadParameter`` for any other mode."""
    import torch

    from multimodal_doc_qa.embed.encoder import MultiVectorEncoder
    from multimodal_doc_qa.index.maxsim import MultiVectorIndex
    from multimodal_doc_qa.retrievers.multivector import MultiVectorRetriever
    from multimodal_doc_qa.retrievers.text import MultiDocTextRetriever, OcrEmbedder, TextRetriever

    _, vision_path, ocr_path = artifact_paths(settings.artifacts_dir)

    if mode == "vision":
        if not vision_path.is_file():
            raise IndexNotFoundError(
                f"no vision index at {vision_path} -- run `doc-qa ingest` first"
            )
        return MultiVectorRetriever(
            encoder=MultiVectorEncoder(settings.embedder_model, settings),
            index=MultiVectorIndex.load(vision_path),
            k=settings.top_k,
        )
    if mode == "ocr":
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
        )
    raise typer.BadParameter(f"mode must be vision or ocr, got {mode!r}")


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
    return GraphDeps(
        retriever=retriever,
        planner_model=chat.with_structured_output(Subqueries),
        assessor_model=chat.with_structured_output(Followups),
        verifier_model=chat.with_structured_output(Unsupported),
        synth=AnswerSynthesizer(settings),
        render_dir=artifact_paths(settings.artifacts_dir)[0],
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

    out = build_graph(deps, settings).invoke(initial_state(question))
    answer = Answer.model_validate(out["answer"]) if out["answer"] else None
    _render_turn(answer, out, budget, settings, settings.artifacts_dir, _ocr_page_texts(settings))
    if answer is None:
        raise typer.Exit(code=1)


@app.command("eval")
def eval_questions(
    questions: Path = typer.Option(
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
    import json

    from multimodal_doc_qa.eval.run import RESULTS_PATH, QuestionRun, run_eval
    from multimodal_doc_qa.schemas import Answer, Question

    settings = Settings()
    source = questions or (
        Path(__file__).resolve().parent / "corpus" / "_artifacts" / "questions.json"
    )
    if not source.is_file():
        ui.render_error(f"no questions at {source}")
        raise typer.Exit(code=1)

    items = [Question.model_validate(q) for q in json.loads(source.read_text())]
    retriever = _open_retriever(settings, mode)

    def run_question(question: Question) -> QuestionRun:
        budget = Budget.from_settings(settings)
        deps = _load_deps(settings, retriever, budget)
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

    summary = run_eval(
        items,
        run_question,
        out_path=(out or RESULTS_PATH),
        settings=settings,
        mode=mode,
        k=settings.top_k,
        iou_threshold=iou_threshold,
        max_tokens=max_tokens,
        max_seconds=max_seconds,
    )

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
    ui.echo(f"written  {out or RESULTS_PATH}")


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
    loaded = {
        path: _open_retriever(settings, path) for path in (mode, ui.other_mode(mode))
    }
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
        try:
            question = prompt.ask().strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not question:
            continue
        if question in {":q", "quit", "exit"}:
            return

        try:
            budget = Budget.from_settings(settings)
            deps = _load_deps(settings, loaded[active["mode"]], budget)
            out = build_graph(deps, settings).invoke(initial_state(question, history))
        except KeyboardInterrupt:
            ui.render_error("turn cancelled")
            continue
        except Exception as exc:  # noqa: BLE001 - one bad turn must not end the session
            ui.render_error(str(exc))
            continue

        answer = Answer.model_validate(out["answer"]) if out["answer"] else None
        _render_turn(answer, out, budget, settings, artifacts, page_texts)
        if answer is not None:
            history.extend(
                [HumanMessage(content=question), AIMessage(content=answer.text)]
            )


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
