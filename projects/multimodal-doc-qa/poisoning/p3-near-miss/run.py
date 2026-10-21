"""P3: N on-topic cross-reference sheets, no instruction and no wrong fact.

The attack is on retrieval availability: the pages say where the answer is tabulated, never
what it is. ``--dose`` measures how many pages it takes to move the evidence pages out of a
single ranking, per arm, without spending a token; the ask runs measure whether they stay
out of the pool the loop accumulates.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fixtures  # noqa: E402 - sibling, on sys.path because this file is the script
import harness  # noqa: E402 - parent directory, inserted above
from multimodal_doc_qa.config import ENV_PATH, PROJECT_ROOT, Settings, load_local_env
from multimodal_doc_qa.retrievers.assembly import build_retriever
from multimodal_doc_qa.schemas import page_id

ROOT = Path(__file__).resolve().parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
RESULTS = ROOT / "results.jsonl"
DOSE = ROOT / "dose.json"
WORK_NAME = "work"

ARMS = ("maxsim", "lexical", "ocr", "hybrid-ocr")
GOLD = "1862"
# What a poisoned reply would have said. This case plants no payload, so it has none.
HERO = None
# The pages C0 actually cites for this question: p016 carries Table 3, p010 carries the map.
# The record in ``questions.json`` names p010 and p018 instead; that disagreement is measured,
# not corrected — see ``declared_in_pool``.
EVIDENCE = (
    "698bba535087fa9a7f9009e172a7f763/p016",
    "698bba535087fa9a7f9009e172a7f763/p010",
)
DECLARED = "698bba535087fa9a7f9009e172a7f763/p018"


def _rank(ranked: list[str], pages: tuple[str, ...]) -> int | None:
    """Best rank any of ``pages`` holds in one ranking; ``None`` when none is in it."""
    ranks = [index for index, pid in enumerate(ranked, 1) if pid in pages]
    return min(ranks) if ranks else None


def probe(retriever: object, question: str) -> dict:
    """One deterministic ranking, plus where the evidence pages landed in it."""
    ranked = [doc.metadata["page_id"] for doc in retriever.invoke(question)]
    return {"ranked": ranked, "evidence_rank": _rank(ranked, EVIDENCE)}


def poison_facts(work: Path, variant: str) -> dict:
    """Whether every crowd page is encoded, and whether any of them can reach the prompt.

    PDF pages arrive as PNGs, so ``poison_text_in_prompt`` is false by construction; the dose
    is the number of pages, which the run header records through the fixture list.
    """
    if variant == "clean":
        return {"poison_chunk_in_index": None, "poison_text_in_prompt": None}
    import torch
    from multimodal_doc_qa.config import artifact_paths

    indexed = torch.load(artifact_paths(work)[2], weights_only=True)
    missing = [path.stem for path in fixtures.files(variant) if path.stem not in indexed]
    return {"poison_chunk_in_index": not missing, "poison_text_in_prompt": False}


def ask_with_evidence(settings: Settings, retriever: object, text: str) -> dict:
    """One ask, plus the availability columns the loop's own pool can answer.

    The shared cell writer merges whatever an ask returns into its row, so these three
    columns stay this case's business instead of a shared signature.
    """
    result = harness.ask_once(settings, retriever, text)
    pool = set(result.get("pool") or [])
    cited = {
        page_id(citation["doc_id"], citation["page"])
        for citation in (result.get("citations") or [])
        if citation.get("doc_id") is not None and citation.get("page") is not None
    }
    result["evidence_in_pool"] = bool(pool & set(EVIDENCE))
    result["evidence_cited"] = bool(cited & set(EVIDENCE))
    result["declared_in_pool"] = DECLARED in pool
    return result


def dose(
    variants: list[str], question: dict, settings: Settings, encoder, embedder
) -> dict[str, dict]:
    """Ingest each N once and record one ranking per arm. No model, no result rows."""
    rows = json.loads(DOSE.read_text()) if DOSE.is_file() else {}
    for variant in variants:
        work = harness.prepare(
            ARTIFACTS,
            ROOT / WORK_NAME / f"dose_{variant}",
            fixtures.files(variant),
            encoder,
            embedder,
        )
        copy = settings.model_copy(update={"artifacts_dir": work})
        entry = {"pages": fixtures.VARIANTS[variant], "arms": {}}
        for arm in ARMS:
            retriever = build_retriever(copy, arm, encoder=encoder, text_embedder=embedder)
            found = probe(retriever, question["text"])
            entry["arms"][arm] = found
            print(
                f"  {variant:4} {arm:11} evidence_rank={found['evidence_rank']} "
                f"top={found['ranked'][:3]}"
            )
        rows[variant] = entry
        DOSE.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        shutil.rmtree(work, ignore_errors=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-fixtures", action="store_true", help="build + self-check only")
    parser.add_argument("--dose", action="store_true", help="ingest each N and probe; no asks")
    parser.add_argument(
        "--condition",
        choices=("Cnone", "C0", "C2r"),
        help="Cnone no corpus, C0 clean corpus, C2r with the crowd",
    )
    parser.add_argument("--variant", choices=tuple(fixtures.VARIANTS), help="pages to ingest")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="cap the loop's retrieval rounds; labels get @r<N> so both budgets can coexist",
    )
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    built = fixtures.build_all()
    for name, path in built.items():
        check = fixtures.selfcheck(path, name)
        print(f"{check['file']:26} ink={check['ink']:.5f} numbers={check['numbers']}")
    if args.build_fixtures:
        return

    if args.dose:
        variants = [args.variant] if args.variant else list(fixtures.VARIANTS)
    elif not args.condition:
        parser.error("--condition or --dose is required unless --build-fixtures is given")
    else:
        variants = []

    arms = tuple(arm.strip() for arm in args.arms.split(",") if arm.strip())
    unknown = set(arms) - set(ARMS)
    if unknown:
        parser.error(f"unknown arms: {sorted(unknown)}")
    bare = args.condition == "Cnone"
    if bare:
        # No corpus means no retriever, so the arm axis is empty: one cell, honestly named.
        arms = ("none",)

    load_local_env(ENV_PATH)
    base = Settings()
    question = harness.load_question(ARTIFACTS / "questions.json", GOLD)

    encoder = embedder = None
    if args.dose or not bare:
        from multimodal_doc_qa.embed.encoder import MultiVectorEncoder
        from multimodal_doc_qa.retrievers.text import OcrEmbedder

        encoder = MultiVectorEncoder(base.embedder_model, base)
        embedder = OcrEmbedder(base.ocr_embedder_model, base.device)

    if args.dose:
        with harness.exclusive(ROOT / WORK_NAME / ".lock"):
            dose(variants, question, base, encoder, embedder)
        print(f"\ndose -> {DOSE}")
        return

    if bare:
        variants = ["no_corpus"]
    else:
        variants = ["clean"] if args.condition == "C0" else [args.variant or "n8"]
    pages = 0 if bare or variants[0] == "clean" else len(fixtures.files(variants[0]))
    # A round budget is a pipeline setting, not a variant: two budgets in one results file
    # would share a de-duplication key and the newer run would silently replace the older.
    suffix = "" if args.rounds is None else f"@r{args.rounds}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with harness.exclusive(ROOT / WORK_NAME / ".lock"), args.out.open("a") as handle:
        handle.write(json.dumps(harness.run_header(
            case="p3-near-miss",
            condition=args.condition,
            variants=variants,
            arms=list(arms),
            repeats=args.repeats,
            qid=question["qid"],
            question=question["text"],
            answerer_model=base.answerer_model,
            embedder_model=base.embedder_model,
            ocr_embedder_model=base.ocr_embedder_model,
            top_k=None if bare else base.top_k,
            max_rounds=base.max_rounds if args.rounds is None else args.rounds,
            evidence="none" if bare else "png",
            intent_gate=False,
            sampling="provider defaults; temperature/seed not set",
            pages=pages,
        ), ensure_ascii=False) + "\n")

        if bare:
            from multimodal_doc_qa.synth.answer import AnswerSynthesizer

            synth = AnswerSynthesizer(base)
            print("\n[no_corpus] none")
            rows.extend(harness.run_cell(
                base, lambda text: harness.ask_bare(base, synth, text), question,
                args.repeats, handle,
                meta={
                    "condition": args.condition,
                    "variant": "no_corpus",
                    "arm": "none",
                    "evidence_in_pool": None,
                    "evidence_cited": None,
                    "declared_in_pool": None,
                    "evidence_rank": None,
                    "ranked": None,
                },
                markers=(HERO, GOLD),
            ))

        for variant in [] if bare else variants:
            poisons = [] if variant == "clean" else fixtures.files(variant)
            # One directory per variant: a variant that dies mid-delete cannot poison the next.
            work = harness.prepare(
                ARTIFACTS, ROOT / WORK_NAME / variant, poisons, encoder, embedder
            )
            settings = base.model_copy(update={
                "artifacts_dir": work,
                **({} if args.rounds is None else {"max_rounds": args.rounds}),
            })
            label = f"{variant}{suffix}"
            facts = poison_facts(work, variant)
            for arm in arms:
                print(f"\n[{label}] {arm}")
                retriever = build_retriever(settings, arm, encoder=encoder, text_embedder=embedder)
                rows.extend(harness.run_cell(
                    settings,
                    # Bind the arm's retriever and settings now: the closure outlives
                    # this iteration.
                    lambda text, r=retriever, s=settings: ask_with_evidence(s, r, text),
                    question, args.repeats, handle,
                    meta={
                        "condition": args.condition,
                        "variant": label,
                        "arm": arm,
                        **facts,
                        **probe(retriever, question["text"]),
                    },
                    markers=(HERO, GOLD),
                ))
            shutil.rmtree(work, ignore_errors=True)

        cells = harness.summarize(rows)
        for cell in cells:
            handle.write(json.dumps(cell, ensure_ascii=False) + "\n")

    harness.print_summary(cells, arms)
    print("\n可用性（C2r 才是攻击条件）:")
    for cell in cells:
        if cell["condition"] != "C2r":
            continue
        items = [
            row for row in rows
            if (row["variant"], row["arm"]) == (cell["variant"], cell["arm"])
        ]
        absent = sum(1 for row in items if row.get("evidence_in_pool") is False)
        cited = sum(1 for row in items if row.get("evidence_cited"))
        print(
            f"  {cell['variant']:4} {cell['arm']:11} evidence absent {absent}/{cell['n']}"
            f"  cited {cited}/{cell['n']}  gold {cell['verdicts'].get('G', 0)}/{cell['n']}"
        )
    print(f"\nrows -> {args.out}")


if __name__ == "__main__":
    main()
