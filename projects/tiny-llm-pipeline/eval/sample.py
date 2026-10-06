"""Samples from the pretrain / sft / dpo checkpoints, side by side.

Pretrain is a base model, so its prompts are raw continuations. SFT and DPO are
chat models, so theirs go through the chat template with a trailing
`<|assistant|>`.

Each prompt is one conversation: a fresh `idx` per call, no history carried
over. Decoding is either greedy (`--temperature 0`) or sampled
(`--temperature` / `--top-k` / `--top-p`). Under sampling every generation gets
a seed derived from (run seed, stage, rep_pen, prompt, sample index), so a reply
is reproducible regardless of the order prompts are run in. Repeated samples
differ only by that index.

`--rep-pen` takes a comma-separated list; each value is reported with three
numbers, so a coefficient can be picked from a curve instead of by eye:

  chars     length of the reply, capped by `--max-new`
  dist4     distinct 4-grams / total 4-grams. 1.0 is rich, low means looping.
  tempo     occurrences of the training template opener ("我用最直白").
            DPO over-fits this prefix; a good coefficient keeps it low.

Caution: `dist4` saturates around 1.2 while the template tail keeps collapsing
to meaningless `最X` stacks up to about 1.4, so the metric alone underpicks.
Check the text too. 1.5 is the delivered value; see PROJECT.md.

Measured 2026-10-06 (6 prompts x 3 samples, rep_pen 1.5): **greedy is the right
protocol for comparing these checkpoints.** Sampling added no content anywhere
and cost coherence; at `temperature=0.8` over the full distribution the DPO
checkpoint emitted 14 U+FFFD in 18 replies, i.e. byte-level BPE picking partial
UTF-8 sequences off the tail. `top_k=40 / top_p=0.9` cut that to 2, and
`temperature=0.5` was the same, but the replies stayed template-plus-filler (DPO)
or off-topic (SFT). `dist4` also stops discriminating: it reads 1.00 for every
sampled reply, so it cannot rank coefficients off the greedy path. Defaults below
sample; pass `--temperature 0` to compare checkpoints.

`--html` writes the run as a self-contained page: one block per prompt, the
stages next to each other, each sample in its own `<pre>`. Readings are per
prompt, so the page is grouped that way instead of by stage.
"""

import argparse
import hashlib
import html
from pathlib import Path

import torch

from tiny_llm_pipeline.model import load_ckpt
from tiny_llm_pipeline.tokenizer import Tok, encode_chat
from tiny_llm_pipeline.train.pretrain import _device

BUNDLE = Path(__file__).resolve().parents[1] / "checkpoints"
TEMPLATE = "我用最直白"
PROMPTS = [
    "你好，介绍一下你自己。",
    "用一句话解释什么是量子纠缠。",
    "我明天要去北京出差，帮我定个行程。",
]

_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>cyber samples</title>
<style>
  body {{ font: 15px/1.6 -apple-system, "PingFang SC", sans-serif; margin: 32px auto;
         max-width: 1280px; color: #1a1a1a; background: #fafafa; }}
  h1 {{ font-size: 20px; }}
  h2 {{ font-size: 17px; border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-top: 36px; }}
  .meta {{ color: #666; font-size: 13px; }}
  .question {{ margin: 28px 0 8px; font-weight: 600; }}
  .cards {{ display: grid; grid-template-columns: repeat({columns}, 1fr); gap: 12px; }}
  .card {{ background: #fff; border: 1px solid #e3e3e3; border-radius: 6px; padding: 12px; }}
  .card h3 {{ margin: 0 0 6px; font-size: 13px; color: #0b5cad; font-weight: 600; }}
  .stats {{ color: #999; font-size: 12px; margin: 8px 0 2px; }}
  pre {{ margin: 0; white-space: pre-wrap; word-wrap: break-word; font: inherit; }}
</style>
<h1>pretrain / sft / dpo — replies</h1>
<p class="meta">{provenance}</p>
"""


def _seed(*parts: object) -> int:
    """A stable 32-bit seed for this generation. Independent of run order."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _stats(text: str) -> dict[str, object]:
    grams = [text[i : i + 4] for i in range(max(0, len(text) - 3))]
    return {
        "chars": len(text),
        "dist4": len(set(grams)) / max(1, len(grams)),
        "tempo": text.count(TEMPLATE),
    }


def _reply(model, stage: str, prompt: str, tok: Tok, end_id: int, device, args, index: int) -> str:
    if stage == "pretrain":
        ids = tok.encode(prompt)
    else:
        ids, _ = encode_chat([{"role": "user", "content": prompt}], tok, add_assistant=True)
    torch.manual_seed(_seed(args.seed, stage, args.rep_pen, prompt, index))
    with torch.no_grad():
        out = model.generate(
            torch.tensor([ids], dtype=torch.long, device=device),
            args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_pen,
        )
    new = out[0, len(ids) :].tolist()
    if end_id in new:
        new = new[: new.index(end_id) + 1]
    return tok.decode(new).strip()


def _samples_for(model, stage, prompts, tok, end_id, device, args) -> list[list[str]]:
    """One list of `--samples` replies per prompt."""
    return [
        [_reply(model, stage, prompt, tok, end_id, device, args, index) for index in range(args.samples)]
        for prompt in prompts
    ]


def _render(prompts, stages, samples, steps, provenance) -> str:
    parts = [_PAGE.format(columns=len(stages), provenance=html.escape(provenance))]
    parts.append(f"<h2>rep_pen={samples['penalty']}</h2>")
    for index, prompt in enumerate(prompts):
        parts.append(f'<div class="question">{index + 1}. {html.escape(prompt)}</div>')
        parts.append('<div class="cards">')
        for stage in stages:
            cards = samples[(stage,)][index]
            body = []
            for position, text in enumerate(cards):
                stats = _stats(text)
                label = f"#{position + 1} " if len(cards) > 1 else ""
                body.append(
                    f'<div class="stats">{label}chars={stats["chars"]} '
                    f'dist4={stats["dist4"]:.2f} tempo={stats["tempo"]}</div>'
                    f"<pre>{html.escape(text)}</pre>"
                )
            parts.append(
                f'<div class="card"><h3>{html.escape(stage)} (step {steps[stage]})</h3>'
                + "".join(body)
                + "</div>"
            )
        parts.append("</div>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--stages", default="pretrain,sft,dpo")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--rep-pen", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 is greedy")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true", help="do not echo replies to stdout")
    parser.add_argument("--html", type=Path, help="archive the run as a self-contained page")
    args = parser.parse_args()

    prompts = args.prompts or PROMPTS
    stages = args.stages.split(",")
    tok = Tok(args.bundle / "tokenizer" / "tokenizer.json")
    end_id = tok.encode("<|end|>")[0]
    device = _device()
    mode = "greedy" if args.temperature <= 0 else (
        f"temperature={args.temperature} top_k={args.top_k} top_p={args.top_p}"
    )
    print(f"device={device.type} tokenizer={tok.hash} samples={args.samples} {mode}")

    steps: dict[str, int] = {}
    samples: dict[tuple[str], list[list[str]]] = {}
    for stage in stages:
        loaded = load_ckpt(args.bundle / stage / "ckpt.pt", expect_tokenizer_hash=tok.hash)
        model = loaded["model"].to(device).eval()
        steps[stage] = loaded["step"]
        print(f"\n===== {stage} (step {loaded['step']}) =====")
        samples[(stage,)] = _samples_for(model, stage, prompts, tok, end_id, device, args)
        for prompt, replies in zip(prompts, samples[(stage,)], strict=True):
            for position, text in enumerate(replies):
                stats = _stats(text)
                print(
                    f"[rp={args.rep_pen} #{position + 1} chars={stats['chars']:3d} "
                    f"dist4={stats['dist4']:.2f} tempo={stats['tempo']}] {prompt}"
                )
                if not args.quiet:
                    print(text + "\n")

    if args.html:
        samples["penalty"] = args.rep_pen
        provenance = (
            f"device={device.type} · tokenizer={tok.hash} (bundle) · rep_pen={args.rep_pen} · "
            f"max_new={args.max_new} · {mode} · {args.samples} sample(s) per prompt · "
            f"single-turn, no history · {len(prompts)} prompts"
        )
        args.html.write_text(_render(prompts, stages, samples, steps, provenance), encoding="utf-8")
        print(f"archived: {args.html}")


if __name__ == "__main__":
    main()
