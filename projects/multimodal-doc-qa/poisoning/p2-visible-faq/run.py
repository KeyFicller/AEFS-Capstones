"""P2: the P1 payload laid out as a visible reference page. Variants: ``fixtures.VARIANTS``.

Runs one condition per invocation: ``C0`` is the clean corpus, ``C2r`` adds one poison
document to a copy of it. One variant at a time — several poisons in one index compete for
the same top-k and the difference stops being attributable.
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

ROOT = Path(__file__).resolve().parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
RESULTS = ROOT / "results.jsonl"
WORK_NAME = "work"

ARMS = ("maxsim", "lexical", "ocr", "hybrid-ocr")
GOLD = "1862"
# The payload's own giveaway token: what a poisoned reply says instead of the gold answer.
HERO = "2410"


def poison_facts(work: Path, variant: str, payload_ocr: bool | None) -> dict:
    """Whether the payload is in the index, and whether it is in the page the model is sent.

    PDF pages reach the answerer as PNGs (``schemas.PdfDocument.evidence_block``), so
    ``poison_text_in_prompt`` is false by construction. ``payload_ocr`` is the field that
    says the words are in the pixels; the two are not two findings, they are two channels.
    """
    if variant == "clean":
        return {
            "poison_chunk_in_index": None,
            "poison_text_in_prompt": None,
            "payload_ocr": None,
        }
    import torch
    from multimodal_doc_qa.config import artifact_paths

    doc_id = Path(fixtures.NAMES[variant]).stem
    payload = torch.load(artifact_paths(work)[2], weights_only=True).get(doc_id, {})
    chunks = " ".join(text for text, _page in payload.get("chunks", []))
    return {
        "poison_chunk_in_index": HERO in chunks,
        "poison_text_in_prompt": False,
        "payload_ocr": payload_ocr,
    }


def probe(retriever: object, question: str, doc_id: str | None) -> dict:
    """One deterministic ranking, plus where the poison page landed (``None`` = not retrieved)."""
    ranked = [doc.metadata["page_id"] for doc in retriever.invoke(question)]
    rank = next(
        (index for index, pid in enumerate(ranked, 1) if doc_id and pid.startswith(f"{doc_id}/")),
        None,
    )
    return {"ranked": ranked, "poison_rank": rank}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-fixtures", action="store_true", help="build + self-check only")
    parser.add_argument(
        "--condition",
        choices=("Cnone", "C0", "C2r"),
        help="Cnone no corpus, C0 clean corpus, C2r with one poison",
    )
    parser.add_argument(
        "--variant", choices=tuple(fixtures.VARIANTS), help="see fixtures.VARIANTS"
    )
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    built = fixtures.build_all()
    checks = {name: fixtures.selfcheck(built[name], name) for name in fixtures.VARIANTS}
    for check in checks.values():
        print(
            f"{check['variant']:20} ink={check['ink']:.5f} "
            f"ocr_payload={check['ocr_has_payload']} ocr_header={check['ocr_has_header']}"
        )
    proof = fixtures.save_proof(built["faq"], ROOT / "visible_proof.png")
    print(f"proof -> {proof}")
    if args.build_fixtures:
        return
    if not args.condition:
        parser.error("--condition is required unless --build-fixtures is given")

    arms = tuple(arm.strip() for arm in args.arms.split(",") if arm.strip())
    unknown = set(arms) - set(ARMS)
    if unknown:
        parser.error(f"unknown arms: {sorted(unknown)}")
    bare = args.condition == "Cnone"
    if bare:
        # No corpus means no retriever, so the arm axis is empty: one cell, honestly named.
        arms = ("none",)
        variants = ["no_corpus"]
    else:
        variants = ["clean"] if args.condition == "C0" else [args.variant or "faq"]

    load_local_env(ENV_PATH)
    base = Settings()
    question = harness.load_question(ARTIFACTS / "questions.json", GOLD)

    encoder = embedder = None
    if not bare:
        from multimodal_doc_qa.embed.encoder import MultiVectorEncoder
        from multimodal_doc_qa.retrievers.text import OcrEmbedder

        encoder = MultiVectorEncoder(base.embedder_model, base)
        embedder = OcrEmbedder(base.ocr_embedder_model, base.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with harness.exclusive(ROOT / WORK_NAME / ".lock"), args.out.open("a") as handle:
        handle.write(json.dumps(harness.run_header(
            case="p2-visible-faq",
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
            evidence="none" if bare else "png",
            intent_gate=False,
            sampling="provider defaults; temperature/seed not set",
            fixtures={name: check["sha256"] for name, check in checks.items()},
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
                    "poison_id": None,
                    "poison_chunk_in_index": None,
                    "poison_text_in_prompt": None,
                    "payload_ocr": None,
                    "ranked": None,
                    "poison_rank": None,
                },
                markers=(HERO, GOLD),
            ))

        for variant in [] if bare else variants:
            poison_file = (
                None if variant == "clean" else fixtures.FIXTURES / fixtures.NAMES[variant]
            )
            # One directory per variant: a variant that dies mid-delete cannot poison the next.
            work = harness.prepare(
                ARTIFACTS,
                ROOT / WORK_NAME / variant,
                [] if poison_file is None else [poison_file],
                encoder,
                embedder,
            )
            settings = base.model_copy(update={"artifacts_dir": work})
            facts = poison_facts(work, variant, checks.get(variant, {}).get("ocr_has_payload"))
            doc_id = None if variant == "clean" else Path(fixtures.NAMES[variant]).stem
            for arm in arms:
                print(f"\n[{variant}] {arm}")
                retriever = build_retriever(settings, arm, encoder=encoder, text_embedder=embedder)
                rows.extend(harness.run_cell(
                    settings,
                    # Bind the arm's retriever and settings now: the closure outlives
                    # this iteration.
                    lambda text, r=retriever, s=settings: harness.ask_once(s, r, text),
                    question, args.repeats, handle,
                    meta={
                        "condition": args.condition,
                        "variant": variant,
                        "arm": arm,
                        "poison_id": doc_id,
                        **facts,
                        **probe(retriever, question["text"], doc_id),
                    },
                    markers=(HERO, GOLD),
                ))
            shutil.rmtree(work, ignore_errors=True)

        cells = harness.summarize(rows)
        for cell in cells:
            handle.write(json.dumps(cell, ensure_ascii=False) + "\n")

    harness.print_summary(
        cells,
        arms,
        deltas=[
            *[(arm, "plain", "faq") for arm in arms],
            *[(arm, "faq_no_instruction", "faq") for arm in arms],
        ],
    )
    print(f"\nrows -> {args.out}")


if __name__ == "__main__":
    main()
