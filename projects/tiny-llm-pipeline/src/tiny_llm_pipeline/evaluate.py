"""Deterministic eval on frozen prompts: text metrics and `results.jsonl`.

Three row kinds, the same shape as the repository's other projects: a `run`
header with provenance, one `item` per (prompt, stage), and a `summary`. Every
row is flushed as it is written, so a run that dies half way keeps what it
already paid for; the file is rewritten from scratch on each run instead of
appended, so two runs can never mix their steps.

No judge model: decoding is greedy and every reading comes from the reply text,
so the same checkpoints always produce the same file. The style lexicon below
follows `artifacts/prefs/doubao-style-guide.md`; the degeneration guardrails
(`repeat_4`, `longest_run`) exist so "more 宝子" cannot be read as alignment.
"""

import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from tiny_llm_pipeline.generate import Bundle, GenSettings
from tiny_llm_pipeline.train.sft import _response_ppl

# Style markers and weights, distilled from the 20-item guide. Soft particles
# are weighted low because they occur in ordinary Chinese too; the address term
# and the "最…" preamble are the identifying features, so they weigh most.
DOUBAO_MARKERS: dict[str, float] = {
    "宝子": 3.0,
    "宝宝": 3.0,
    "亲爱的": 2.0,
    "哒": 2.0,
    "嘻嘻": 2.0,
    "嘿嘿": 2.0,
    "本豆": 2.0,
    "豆包": 1.5,
    "太懂你": 2.0,
    "完全正确": 2.0,
    "我错了": 2.0,
    "我在这儿": 2.0,
    "陪你": 1.0,
    "稳稳": 1.0,
    "～": 1.0,
    "~": 0.5,
    "呢": 1.0,
    "嘛": 0.5,
    "啦": 0.5,
    "哦": 0.5,
    "呀": 0.5,
}


def doubao_score(text: str) -> float:
    """Weighted style markers per 100 characters. Higher means more 豆包体."""
    hits = sum(weight * text.count(marker) for marker, weight in DOUBAO_MARKERS.items())
    return hits / max(len(text), 1) * 100


def distinct_n(texts: list[str], n: int) -> float:
    """Unique n-grams over total n-grams across `texts`. 0.0 when there are none."""
    grams = [text[i : i + n] for text in texts for i in range(max(0, len(text) - n + 1))]
    return len(set(grams)) / len(grams) if grams else 0.0


def repeat_4(text: str) -> float:
    """Share of a single reply's 4-grams that are repeats. 1 - distinct-4."""
    return 1.0 - distinct_n([text], 4)


def longest_run(text: str) -> int:
    """Longest run of one repeated character, e.g. four exclamation marks reads 4."""
    best = run = 0
    previous = ""
    for char in text:
        run = run + 1 if char == previous else 1
        previous = char
        best = max(best, run)
    return best


def evaluate(
    bundle: Bundle,
    prompts: list[str],
    settings: GenSettings,
    out: Path | str,
    *,
    stages: tuple[str, ...] = ("sft", "dpo"),
    holdout: list[dict] | None = None,
) -> dict:
    """Generate, score, and write `results.jsonl`. Returns the summary row.

    `holdout` is the `sft_holdout.jsonl` rows, used for response-token
    perplexity per stage with the same template and mask SFT trained with.
    """
    if not prompts:
        raise ValueError("no eval prompts")
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    items: list[dict] = []
    with dest.open("w", encoding="utf-8") as handle:
        _write(handle, _run_row(bundle, prompts, settings, stages, holdout))
        for stage in stages:
            generator = bundle.generators[stage]
            for prompt in prompts:
                reply = generator.reply(prompt, settings)
                item = {
                    "kind": "item",
                    "prompt": prompt,
                    "stage": stage,
                    "step": generator.step,
                    "reply": reply,
                    "chars": len(reply),
                    "doubao_score": round(doubao_score(reply), 3),
                    "distinct_1": round(distinct_n([reply], 1), 4),
                    "distinct_2": round(distinct_n([reply], 2), 4),
                    "repeat_4": round(repeat_4(reply), 4),
                    "longest_run": longest_run(reply),
                }
                items.append(item)
                _write(handle, item)
        summary = _summary(bundle, items, prompts, stages, holdout, started)
        _write(handle, summary)
    return summary


def _write(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def _run_row(
    bundle: Bundle,
    prompts: list[str],
    settings: GenSettings,
    stages: tuple[str, ...],
    holdout: list[dict] | None,
) -> dict:
    digest = hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()[:16]
    return {
        "kind": "run",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": _commit(),
        "device": bundle.device.type,
        "tokenizer": bundle.tok.hash,
        "stages": list(stages),
        "steps": bundle.steps,
        "n_prompts": len(prompts),
        "prompts_sha256": digest,
        "n_holdout": len(holdout) if holdout else 0,
        "decoding": "greedy",
        "rep_pen": settings.rep_pen,
        "max_new": settings.max_new,
        "seed": settings.seed,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }


def _commit() -> str | None:
    """Short HEAD of the repository, or None outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _summary(
    bundle: Bundle,
    items: list[dict],
    prompts: list[str],
    stages: tuple[str, ...],
    holdout: list[dict] | None,
    started: float,
) -> dict:
    """Mean readings per stage, the paired gap, and optional perplexity."""
    summary: dict = {
        "kind": "summary",
        "n_items": len(items),
        "n_prompts": len(prompts),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    for metric in ("chars", "doubao_score", "distinct_1", "distinct_2", "repeat_4", "longest_run"):
        summary[metric] = {
            stage: _mean([item[metric] for item in items if item["stage"] == stage])
            for stage in stages
        }
    # Per-reply `repeat_4` only sees a loop inside one answer. A model that opens
    # every answer with the same phrase repeats across prompts, which this catches.
    summary["cross_prompt_distinct_4"] = {
        stage: round(
            distinct_n([item["reply"] for item in items if item["stage"] == stage], 4), 4
        )
        for stage in stages
    }
    if "sft" in stages and "dpo" in stages:
        scores = {
            stage: [item["doubao_score"] for item in items if item["stage"] == stage]
            for stage in ("sft", "dpo")
        }
        deltas = [dpo - sft for sft, dpo in zip(scores["sft"], scores["dpo"], strict=True)]
        summary["doubao_delta_dpo_minus_sft"] = {
            "mean": _mean(deltas),
            "ci95": _bootstrap_ci(deltas, seed=0),
            "dpo_win_rate": _share(deltas, lambda value: value > 0),
            "tie_rate": _share(deltas, lambda value: value == 0),
            "sft_win_rate": _share(deltas, lambda value: value < 0),
        }
    if holdout:
        summary["ppl"] = {
            stage: _finite(
                _response_ppl(bundle.generators[stage].model, holdout, bundle.tok, bundle.device)
            )
            for stage in stages
        }
    return summary


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _finite(value: float) -> float | None:
    """JSON has no infinity; an unusable perplexity is `null`, not a fake number."""
    return round(value, 4) if value == value and value != float("inf") else None


def _share(values: list[float], predicate) -> float | None:
    if not values:
        return None
    return round(sum(1 for value in values if predicate(value)) / len(values), 4)


def _bootstrap_ci(values: list[float], seed: int, draws: int = 2000) -> list[float] | None:
    """Percentile 95% interval of the mean, by resampling `values`."""
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(draws))
    return [round(means[int(0.025 * draws)], 4), round(means[int(0.975 * draws) - 1], 4)]
