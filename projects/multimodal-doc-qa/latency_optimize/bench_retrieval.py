"""Per-stage latency of the retrieval chain: one-time costs vs per-query costs.

One-time rows (model loads, index loads) print in seconds and are paid once per
process; per-query rows (query encoding, MaxSim search, full ``invoke``) print in
milliseconds and are paid on every retrieve hop. Confusing the two is how a 40 ms
retrieve ends up looking like an 8 s one.

Stages are attributed by synchronising MPS after the forward pass: without that the
GPU launch returns early and its cost is billed to the next stage instead.

The default modes are the classic chain: ``maxsim`` + ``lexical`` (BM25) + their RRF
``hybrid-maxsim``. ``--rerank`` adds the VLM rerank stage; it is one network call per
query, so it is timed once rather than ``--reps`` times.

    ./start.sh run python projects/multimodal-doc-qa/latency_optimize/bench_retrieval.py
    ./start.sh run python projects/multimodal-doc-qa/latency_optimize/bench_retrieval.py \
        --modes maxsim,hybrid-maxsim --reps 20 --limit 10 \
        --json projects/multimodal-doc-qa/latency_optimize/run.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FALLBACK_QUERIES = (
    "Where was Gestalt psychology concieved?",
    "According to the chart on page 14 how much time was spent with family and friends?",
    "How does the method handle long documents?",
)


def time_ms(fn: Callable[[], object], reps: int, warmup: int = 1) -> list[float]:
    """Wall-clock milliseconds per call, after ``warmup`` untimed calls."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def load_queries(path: Path, limit: int) -> list[str]:
    """``limit`` distinct question texts, or the built-in fallbacks when there is no file."""
    if not path.is_file():
        return list(FALLBACK_QUERIES)[:limit]
    rows = json.loads(path.read_text(encoding="utf-8"))
    seen: list[str] = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if text and text not in seen:
            seen.append(text)
        if len(seen) >= limit:
            break
    return seen or list(FALLBACK_QUERIES)[:limit]


def sync(device: str) -> None:
    """Block until queued MPS work is done, so the next stage does not pay for this one."""
    if device == "mps":
        import torch

        torch.mps.synchronize()


def encode_stages(encoder: object, query: str, reps: int) -> dict[str, list[float]]:
    """Split one ``encode_query`` into processor / forward / transfer / cache-drop samples.

    Mirrors ``MultiVectorEncoder.encode_query`` step by step, synchronising after each
    GPU stage. The stages sum to the production path; the split says where it goes.
    """
    import torch

    device = encoder.settings.device
    batch = encoder.processor.process_queries([query]).to(device)
    sync(device)
    with torch.no_grad():
        encoder.model(**batch)
    sync(device)

    samples: dict[str, list[float]] = {
        "processor": [],
        "forward": [],
        "transfer": [],
        "empty_cache": [],
    }
    for _ in range(reps):
        start = time.perf_counter()
        batch = encoder.processor.process_queries([query]).to(device)
        sync(device)
        samples["processor"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        with torch.no_grad():
            embedding = encoder.model(**batch)
        sync(device)
        samples["forward"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        embedding[0].to(torch.float32).cpu()
        samples["transfer"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        if device == "mps":
            torch.mps.empty_cache()
        samples["empty_cache"].append((time.perf_counter() - start) * 1000.0)
    return samples


def search_rows(
    label: str, retriever: object, encoder: object, queries: Sequence[str], reps: int
) -> list[dict[str, float | str]]:
    """Time the ranking step alone, with the query vector already in hand.

    ``invoke`` conflates encoding with ranking; at a few thousand pages the ranking
    step stops being noise, and this is the row that says how much of it is corpus.
    """
    index = getattr(retriever, "index", None)
    if index is not None and encoder is not None:
        samples: list[float] = []
        for query in queries:
            vec = encoder.encode_query(query)  # held out of the timer on purpose
            samples.extend(time_ms(lambda i=index, v=vec, k=retriever.k: i.search(v, k=k), reps))
        return [summarize(f"{label}: index.search", samples)]

    bm25 = getattr(retriever, "_bm25", None)
    if bm25 is not None:
        from multimodal_doc_qa.retrievers.bm25 import tokenize

        samples = []
        n_chunks = len(retriever.corpus)
        for query in queries:
            tokens = tokenize(query)
            samples.extend(
                time_ms(
                    lambda t=tokens, b=bm25, n=n_chunks: [
                        b.score(t, position) for position in range(n)
                    ],
                    reps,
                )
            )
        return [summarize(f"{label}: score loop", samples)]
    return []


def corpus_scale(artifacts_dir: Path) -> str:
    """``docs/pages/chunks`` from the ingest manifest, so a run states its own corpus."""
    manifest = artifacts_dir / "manifest.json"
    if not manifest.is_file():
        return "corpus=unknown"
    counts = json.loads(manifest.read_text(encoding="utf-8")).get("counts", {})
    return (
        f"corpus={counts.get('documents', '?')}docs/"
        f"{counts.get('pages', '?')}pages/{counts.get('chunks', '?')}chunks"
    )


def summarize(label: str, samples: Sequence[float]) -> dict[str, float | str]:
    """Median / p90 / min of one stage's samples, ready to print or dump."""
    ordered = sorted(samples)
    return {
        "stage": label,
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))],
        "min": ordered[0],
        "n": len(ordered),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--modes", default="maxsim,lexical,hybrid-maxsim")
    parser.add_argument("--reps", type=int, default=10, help="timed calls per query")
    parser.add_argument("--limit", type=int, default=5, help="distinct bench queries")
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="question file; defaults to <artifacts_dir>/questions.json, so MDQ_ARTIFACTS_DIR is honoured",
    )
    parser.add_argument("--json", type=Path, default=None, help="also write the rows here")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="also time the VLM rerank stage (one network call per query)",
    )
    args = parser.parse_args(argv)

    from multimodal_doc_qa.config import Settings
    from multimodal_doc_qa.index.maxsim import MultiVectorIndex
    from multimodal_doc_qa.retrievers.assembly import (
        ARMS,
        COMPONENTS,
        INDEXES,
        IndexNotFoundError,
        build_retriever,
    )

    settings = Settings()
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    unknown = [mode for mode in modes if mode not in ARMS]
    if unknown:
        raise SystemExit(f"unknown mode(s) {unknown}; expected a subset of {list(ARMS)}")

    stages = {stage for mode in modes for stage in ARMS[mode].stages}
    indexes = {COMPONENTS[stage].index for stage in stages}
    queries = load_queries(args.questions or settings.artifacts_dir / "questions.json", args.limit)

    one_time: list[dict[str, float | str]] = []
    per_query: list[dict[str, float | str]] = []

    print(
        f"device={settings.device} dtype={settings.dtype} top_k={settings.top_k} reps={args.reps}"
    )
    print(f"artifacts={settings.artifacts_dir} {corpus_scale(settings.artifacts_dir)}")
    print(f"modes={','.join(modes)} queries={len(queries)} first={queries[0][:48]!r}\n")

    encoder = None
    if "multivector" in indexes:
        from multimodal_doc_qa.embed.encoder import MultiVectorEncoder

        start = time.perf_counter()
        encoder = MultiVectorEncoder(settings.embedder_model, settings)
        one_time.append(
            {"stage": "load MultiVectorEncoder", "seconds": time.perf_counter() - start}
        )

    text_embedder = None
    if any(
        COMPONENTS[stage].index in {"ocr", "abstract"} and COMPONENTS[stage].family == "embedding"
        for stage in stages
    ):
        from multimodal_doc_qa.retrievers.text import OcrEmbedder

        start = time.perf_counter()
        text_embedder = OcrEmbedder(settings.ocr_embedder_model, settings.device)
        one_time.append({"stage": "load OcrEmbedder", "seconds": time.perf_counter() - start})

    for name in sorted(indexes):
        path = INDEXES[name].path(settings)
        if not path.is_file():
            continue
        start = time.perf_counter()
        MultiVectorIndex.load(path)
        one_time.append({"stage": f"load {name} index", "seconds": time.perf_counter() - start})

    retrievers: dict[str, object] = {}
    for mode in modes:
        start = time.perf_counter()
        try:
            retrievers[mode] = build_retriever(
                settings, mode, encoder=encoder, text_embedder=text_embedder
            )
        except IndexNotFoundError as exc:
            print(f"[skip] {mode}: {exc}")
            continue
        one_time.append({"stage": f"build {mode}", "seconds": time.perf_counter() - start})

    if encoder is not None:
        stage_samples: dict[str, list[float]] = {
            "processor": [],
            "forward": [],
            "transfer": [],
            "empty_cache": [],
        }
        for query in queries:
            for name, values in encode_stages(encoder, query, args.reps).items():
                stage_samples[name].extend(values)
        for name, values in stage_samples.items():
            per_query.append(summarize(f"encode_query/{name}", values))
        per_query.append(
            summarize(
                "encode_query [total]",
                [
                    value
                    for query in queries
                    for value in time_ms(lambda q=query: encoder.encode_query(q), args.reps)
                ],
            )
        )

    for mode, retriever in retrievers.items():
        per_query.append(
            summarize(
                f"{mode}: invoke",
                [
                    value
                    for query in queries
                    for value in time_ms(lambda q=query, r=retriever: r.invoke(q), args.reps)
                ],
            )
        )
        dense = getattr(retriever, "dense", None)
        lexical = getattr(retriever, "lexical", None)
        if dense is None or lexical is None:
            per_query.extend(search_rows(mode, retriever, encoder, queries, args.reps))
            continue
        for label, part in (("dense", dense), ("lexical", lexical)):
            per_query.append(
                summarize(
                    f"{mode}: {label}",
                    [
                        value
                        for query in queries
                        for value in time_ms(lambda q=query, p=part: p.invoke(q), args.reps)
                    ],
                )
            )
            per_query.extend(search_rows(f"{mode}/{label}", part, encoder, queries, args.reps))

    if args.rerank:
        from multimodal_doc_qa.config import artifact_paths, documents_path
        from multimodal_doc_qa.retrievers.rerank import PageRank, rerank_pages
        from multimodal_doc_qa.schemas import load_documents
        from multimodal_doc_qa.synth.answer import build_chat_model

        start = time.perf_counter()
        ranker = build_chat_model(settings).with_structured_output(PageRank)
        one_time.append({"stage": "build rerank model", "seconds": time.perf_counter() - start})

        target = "hybrid-maxsim" if "hybrid-maxsim" in retrievers else next(iter(retrievers), None)
        if target is None:
            print("[skip] rerank: no retriever was built")
        else:
            catalog_path = documents_path(settings.artifacts_dir)
            catalog = load_documents(catalog_path) if catalog_path.is_file() else None
            render_dir = artifact_paths(settings.artifacts_dir)[0]
            samples = []
            for query in queries:
                documents = retrievers[target].invoke(query)
                start = time.perf_counter()
                rerank_pages(ranker, query, documents, render_dir, catalog)
                samples.append((time.perf_counter() - start) * 1000.0)
            per_query.append(summarize(f"rerank: {target} (1 call/query)", samples))

    print("ONE-TIME (seconds, paid once per process)")
    print(f"{'stage':34s}{'sec':>8s}")
    for row in one_time:
        print(f"{row['stage']:34s}{row['seconds']:8.2f}")

    print("\nPER-QUERY (milliseconds over all bench queries)")
    print(f"{'stage':34s}{'median':>9s}{'p90':>9s}{'min':>9s}{'n':>6s}")
    for row in per_query:
        print(
            f"{row['stage']:34s}{row['median']:9.2f}{row['p90']:9.2f}{row['min']:9.2f}{row['n']:6d}"
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "device": settings.device,
            "dtype": settings.dtype,
            "top_k": settings.top_k,
            "reps": args.reps,
            "artifacts_dir": str(settings.artifacts_dir),
            "corpus": corpus_scale(settings.artifacts_dir),
            "n_queries": len(queries),
            "one_time_seconds": one_time,
            "per_query_ms": per_query,
        }
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
