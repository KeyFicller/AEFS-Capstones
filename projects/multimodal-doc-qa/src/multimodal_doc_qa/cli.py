"""CLI: ``ingest`` writes the indices, ``ask`` and ``eval`` read them.

Heavy imports stay inside the commands so ``--help`` does not load torch.
"""

from pathlib import Path

import typer
from rich.console import Console

from multimodal_doc_qa.budget import Budget
from multimodal_doc_qa.config import Settings, artifact_paths
from multimodal_doc_qa.graph import GraphDeps, build_graph, initial_state
from multimodal_doc_qa.schemas import Answer, page_id

app = typer.Typer(help="Agentic RAG over document images", add_completion=False)
console = Console()


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
    from multimodal_doc_qa.retrievers.text import TextEmbedder

    settings = Settings()
    artifacts = (out or settings.artifacts_dir).expanduser()
    render_dir, vision_path, ocr_path = artifact_paths(artifacts)

    pdfs = sorted(corpus.expanduser().glob("*.pdf"))
    if not pdfs:
        console.print(f"[red]no PDFs in {corpus}[/]")
        raise typer.Exit(code=1)

    vision = MultiVectorIndex(device="cpu")
    ocr_docs: dict[str, dict] = {}

    encoder = MultiVectorEncoder(settings.embedder_model, settings)
    console.print(f"encoder   {type(encoder.model).__name__} ({settings.device}/{settings.dtype})")
    text_embedder = TextEmbedder()

    for pdf in pdfs:
        doc_id = pdf.stem
        pages = render_pages(pdf, render_dir / doc_id)

        for page in pages:
            with Image.open(page) as image:
                vectors = encoder.encode_images([image.convert("RGB")])[0]
            vision.add(page_id(doc_id, int(page.stem[1:])), vectors)

        chunks = chunk_texts(extract_page_texts(pdf, pages))
        ocr_docs[doc_id] = {
            "index": text_embedder.encode([text for text, _ in chunks]),
            "chunks": chunks,
        }
        console.print(f"  {doc_id}: {len(pages)} pages, {len(chunks)} text chunks")

    vision.save(vision_path)
    ocr_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ocr_docs, ocr_path)

    console.print(f"vision    {vision_path}  ({vision.nbytes() / 1e6:.1f} MB)")
    console.print(f"ocr       {ocr_path}")
    console.print(f"[green]ingested[/] {len(pdfs)} docs -> {artifacts}")


def _load_retriever(settings: Settings, mode: str):
    """Build the vision or OCR retriever. Raises ``BadParameter`` for any other mode."""
    import torch

    from multimodal_doc_qa.embed.encoder import MultiVectorEncoder
    from multimodal_doc_qa.index.maxsim import MultiVectorIndex
    from multimodal_doc_qa.retrievers.multivector import MultiVectorRetriever
    from multimodal_doc_qa.retrievers.text import MultiDocTextRetriever, TextEmbedder, TextRetriever

    _, vision_path, ocr_path = artifact_paths(settings.artifacts_dir)

    if mode == "vision":
        if not vision_path.is_file():
            console.print(f"[red]no vision index at {vision_path}[/] -- run `doc-qa ingest` first")
            raise typer.Exit(code=1)
        return MultiVectorRetriever(
            encoder=MultiVectorEncoder(settings.embedder_model, settings),
            index=MultiVectorIndex.load(vision_path),
            k=settings.top_k,
        )
    if mode == "ocr":
        if not ocr_path.is_file():
            console.print(f"[red]no OCR index at {ocr_path}[/] -- run `doc-qa ingest` first")
            raise typer.Exit(code=1)
        embedder = TextEmbedder()
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
    mode: str = typer.Option("vision", "--mode", help="vision | ocr"),
) -> None:
    """Answer one question through the agentic graph."""
    settings = Settings()

    retriever = _load_retriever(settings, mode)
    budget = Budget.from_settings(settings)
    deps = _load_deps(settings, retriever, budget)

    out = build_graph(deps, settings).invoke(initial_state(question))

    console.print(f"[bold]{question}[/] ({mode})")
    if out["answer"] is None:
        console.print(f"[yellow]no answer[/] (stop_reason={out['stop_reason']})")
        raise typer.Exit(code=1)

    answer = Answer.model_validate(out["answer"])
    console.print(answer.text)
    for citation in answer.citations:
        console.print(f"  cite {page_id(citation.doc_id, citation.page)}  bbox={citation.bbox}")
    console.print(
        f"rounds {out['rounds']}/{settings.max_rounds}"
        f"  calls {budget.calls}/{settings.max_ask_calls}"
        f"  tokens {budget.tokens}/{settings.max_ask_tokens}"
        f"  {budget.elapsed():.1f}s/{settings.max_ask_seconds:.0f}s"
    )
    if out["stop_reason"]:
        console.print(f"stop_reason {out['stop_reason']}")


@app.command("eval")
def eval_questions(
    questions: Path = typer.Option(
        None, "--questions", help="Holdout JSON (default: the corpus artifacts)"
    ),
    mode: str = typer.Option("vision", "--mode", help="vision | ocr"),
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
        console.print(f"[red]no questions at {source}[/]")
        raise typer.Exit(code=1)

    items = [Question.model_validate(q) for q in json.loads(source.read_text())]
    retriever = _load_retriever(settings, mode)

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

    console.print(f"[bold]{mode}[/] over {summary['n_done']}/{summary['n_questions']} questions")
    console.print(
        f"nDCG@{settings.top_k} {summary['ndcg_at_k']:.4f}"
        f"  IoU@{iou_threshold:g} {summary['iou_at_threshold']:.4f}"
        f"  (strict containment {summary['bbox_hit_rate']:.4f})"
    )
    console.print(
        f"latency p50/p95 {summary['latency_p50_s']:.2f}s/{summary['latency_p95_s']:.2f}s"
        f"  tokens {summary['tokens']}  stop_reasons {summary['stop_reasons']}"
    )
    if summary["stopped_after"]:
        console.print(f"[yellow]cut short before {summary['stopped_after']}[/]")
    console.print(f"written  {out or RESULTS_PATH}")


if __name__ == "__main__":
    app()
