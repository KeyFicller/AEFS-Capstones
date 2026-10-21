"""Synthesize DPO preference pairs: one over-the-top 豆包体 answer, one neutral one.

The chosen system prompt is the operationable distillation of
`artifacts/prefs/doubao-style-guide.md` (the 20-item style outline): the 最…
preamble is capped at 3, reduplication at 2. Output is append-only, so a re-run
skips finished prompts and the API is paid for once.
"""

import hashlib
import json
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from rich import print as rprint

from tiny_llm_pipeline.config import DEFAULT_MODEL
from tiny_llm_pipeline.tokenizer import Tok, encode_chat

CHOSEN_SYS = "用夸张的『豆包体』回答，严格照着下面的风格纲要来。"
REJECTED_SYS = (
    "把用户给出的回答改写成中性、专业、简洁的助理口吻。"
    "保留全部事实、结论和结构，不增删信息，长度与原回答相当。只输出改写后的回答。"
)

# Measured on the paid 822 pairs: 51% of `chosen` carried an 8-char run copied
# verbatim from the outline, the most common 12-char opener appeared 132 times,
# and 28% of pairs shared their opener with at least ten others. DPO learned
# that opener as a rule — the cross-prompt 4-gram diversity of its answers fell
# to 0.572 from SFT's 0.862. The outline's examples are the leak, so the prompt
# has to say they are examples.
_NO_COPY = (
    "纲要里的例句只是示范，不要照抄原句；每条回答换一种开场说法，"
    "不要每条都用同一句，也不要每次都从「宝子」起头。"
)

# A model told to "vary the opening" with no alternative given falls back to
# the outline's own canonical preamble: 132 of the first 822 replies opened
# with it, and the cross-prompt 4-gram diversity of DPO's answers fell to 0.572
# from SFT's 0.862 as a result. Naming one angle per prompt is the alternative
# `_NO_COPY` cannot supply.
_OPENERS = (
    "「最…」排比前摇：先声明自己要用多直接的方式回答",
    "夸张共情开场：先接住用户的情绪，再谈内容",
    "先认同：先说用户说得对，再补上自己的看法",
    "拟人化副语言开场：用迟疑、停顿或语气词起头",
    "过度热情开场：连发短感叹句",
    "陪伴与在场感开场：强调你在这儿稳稳陪着对方",
    "亲密称呼开场：用「宝子 / 宝宝 / 亲爱的」起头",
    "不要开场白：第一句直接给答案，语气词放在句中",
)


def _opener(prompt: str) -> str:
    """One opening angle per prompt, keyed by the prompt text itself.

    Stable across appends and shards, so a resumed run keeps the same rotation
    and two shards never give the same prompt different openers.
    """
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return _OPENERS[int.from_bytes(digest[:4], "big") % len(_OPENERS)]

# The prompt is the only place the two caps are stated: the 最… preamble and the
# reduplication whitelist are both enumerable repetition, so capping them here
# is what keeps `chosen` from degenerating into a chant. The scorer may lag.
# The length target is a number, not an adjective: an earlier "keep it tight"
# wording made the model aim at the ceiling and fill half the context. The token
# figure is `max_len`, the same number `_fits` enforces, so the two cannot drift.
def _budget(max_len: int) -> str:
    return (
        f"回答控制在 160 字以内，整条（问题 + 回答）不超过 {max_len} token。"
        "写之前先挑重点，宁可少说两句，也不要写到一半被截断。"
    )


def chosen_system(guide: str, max_len: int = DEFAULT_MODEL.max_seq_len, opener: str | None = None) -> str:
    """The style guide verbatim, plus the no-copy note, an opening angle, and the budget."""
    parts = [CHOSEN_SYS, guide.strip(), _NO_COPY]
    if opener is not None:
        parts.append(f"这次只用这一种开场方向：{opener}。其它开场方向这次都不要用。")
    parts.append(_budget(max_len))
    return "\n\n".join(parts)


# A refusal names its own inability at the very start. Matching these words
# anywhere in the reply drops valid pairs: 无法 / 不能 / 抱歉 are ordinary
# Chinese, and the rerun of the 168 dropped prompts showed 仅凭这一句话无法判断
# — a normal answer — being thrown away for a 无法 inside the sentence.
_APOLOGY = ("抱歉", "对不起", "很遗憾", "恕难", "不便")
_UNABLE = re.compile(r"^(目前|暂时|现在|我|本豆|这里)?\s*(无法|不能|不便|不会)")
DEFAULT_MODEL_ID = "deepseek-v4-flash"
_MAX_TRIES = 5


def is_rejected(text: str) -> bool:
    """True when the reply opens as a refusal, so the pair must be dropped."""
    head = text.lstrip()
    return head.startswith(_APOLOGY) or bool(_UNABLE.match(head))


def synth_one(prompt: str, llm, chosen_sys: str = CHOSEN_SYS) -> tuple[str, str]:
    """Return `(chosen, rejected)`: one answer, two voices.

    The rejected side is a neutral rewrite of `chosen`, so the two sides carry
    the same content and roughly the same length and only the style differs.
    """
    chosen = _invoke(llm, chosen_sys, prompt)
    rejected = _invoke(llm, REJECTED_SYS, prompt, answer=chosen)
    return chosen, rejected


def synth_prefs(
    prompts: list[str],
    out_jsonl: Path | str,
    *,
    n: int,
    tok: Tok,
    guide: str | None = None,
    max_len: int = DEFAULT_MODEL.max_seq_len,
    model: str = DEFAULT_MODEL_ID,
    llm=None,
    resume: bool = True,
    sleep: float = 0.0,
) -> Path:
    """Write up to `n` pairs to `out_jsonl` and return the path.

    Only `prompts[:n]` are considered. With `resume=True`, prompts already in
    the file are skipped, so an interrupted run continues where it stopped.
    `guide` is the style outline embedded verbatim into the chosen system
    prompt; `None` falls back to the bare instruction.
    A pair whose encoded `prompt + answer` exceeds `max_len` tokens is dropped
    and counted, never truncated: cutting the tail removes the end-of-reply
    signal and makes half an answer incomparable to the full other side. The
    system prompt is not part of that measurement: only the student's
    `user + assistant` turns have to fit its context.
    A dropped prompt is written to the `<stem>.skipped.jsonl` sidecar next to
    `out_jsonl`, and `resume=True` reads that file too: a refusal is never
    re-paid for on the next run. Use `resume=False` to retry the whole list.
    `sleep` throttles between prompts; retries back off on their own.
    """
    dest = Path(out_jsonl)
    dest.parent.mkdir(parents=True, exist_ok=True)
    skipped = _skipped_path(dest)
    done = _existing_prompts(dest) | _existing_prompts(skipped) if resume else set()
    client = llm if llm is not None else _client(model)
    dropped = 0
    over_limit = 0
    gaps: list[int] = []
    for prompt in prompts[:n]:
        if prompt in done:
            continue
        chosen_sys = (
            chosen_system(guide, max_len, _opener(prompt)) if guide else CHOSEN_SYS
        )
        chosen, rejected = synth_one(prompt, client, chosen_sys)
        reason = _drop(chosen, rejected)
        if reason:
            dropped += 1
            _append(skipped, _skip_record(prompt, reason, model))
            rprint(f"[yellow]dropped[/yellow] {dropped} total: {prompt[:40]}")
            continue
        if not _fits(prompt, chosen, rejected, tok, max_len):
            over_limit += 1
            _append(skipped, _skip_record(prompt, "over_limit", model))
            rprint(f"[yellow]over {max_len} tokens[/yellow] ({over_limit}): {prompt[:40]}")
            continue
        _append(
            dest,
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "model": model,
                "created_at": _now(),
            },
        )
        gaps.append(_chat_len(prompt, chosen, tok) - _chat_len(prompt, rejected, tok))
        done.add(prompt)
        if sleep:
            time.sleep(sleep)
    if over_limit:
        rprint(f"[yellow]{over_limit} pairs dropped for exceeding {max_len} tokens[/yellow]")
    if gaps:
        # The rejected side is a rewrite, so this should sit near zero. A wide
        # spread means the two sides differ in length as well as style, which is
        # what makes DPO learn "shorter" instead of "on-brand".
        median = sorted(gaps)[len(gaps) // 2]
        rprint(f"[cyan]token-length gap chosen−rejected: median {median:+d}, max |gap| {max(map(abs, gaps))}[/cyan]")
    return dest


def _fits(prompt: str, chosen: str, rejected: str, tok: Tok, max_len: int) -> bool:
    """True when both sides fit the model context, measured in real tokens."""
    return _chat_len(prompt, chosen, tok) <= max_len and _chat_len(prompt, rejected, tok) <= max_len


def _chat_len(prompt: str, reply: str, tok: Tok) -> int:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reply},
    ]
    return len(encode_chat(messages, tok)[0])


def _drop(chosen: str, rejected: str) -> str | None:
    """Why this pair is unusable, or None when it can be kept.

    The reason names the side: a refusal on `chosen` means the style prompt
    itself failed, on `rejected` it usually means the prompt asked about text
    that was in an earlier turn. One reason for both sides hid that split.
    """
    if not chosen or not rejected:
        return "empty"
    if chosen == rejected:
        return "identical"
    if is_rejected(chosen):
        return "refusal:chosen"
    if is_rejected(rejected):
        return "refusal:rejected"
    return None


def _invoke(llm, system: str, prompt: str, *, answer: str | None = None) -> str:
    """Call the model, retrying only transient failures.

    Messages are `(role, content)` tuples, not `SystemMessage`, so tests can
    read the system prompt as `msgs[0][1]`. `answer` carries the reply a
    rewrite is meant to preserve; without it the call answers `prompt` itself.
    """
    user = prompt if answer is None else f"问题：{prompt}\n\n要改写的回答：{answer}"
    messages = [("system", system), ("user", user)]
    delay = 1.0
    for attempt in range(_MAX_TRIES):
        try:
            return llm.invoke(messages).content.strip()
        except Exception as exc:
            if attempt == _MAX_TRIES - 1 or not _transient(exc):
                raise
            time.sleep(delay)
            delay *= 2


def _transient(exc: BaseException) -> bool:
    """Timeouts and rate limits are worth a retry; a bad key is not."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "timeout" in name
        or "ratelimit" in name
        or "connection" in name
        or "429" in text
        or "rate limit" in text
        or "timed out" in text
    )


def _existing_prompts(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    prompts: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            prompts.add(str(json.loads(line)["prompt"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return prompts


def copies_guide(text: str, guide: str, span: int = 8) -> bool:
    """True when `text` carries a verbatim run from the style outline.

    The outline is pasted into the prompt, so its example sentences come back
    as replies unless told otherwise: 51% of the paid 822 pairs did, and the
    most common opening appeared 132 times. DPO turned that opening into a rule.
    """
    window = {guide[i : i + span] for i in range(max(0, len(guide) - span + 1))}
    return any(text[i : i + span] in window for i in range(max(0, len(text) - span + 1)))


def write_style_sft(
    prefs_jsonl: Path | str,
    out_dir: Path | str,
    *,
    tok_dir: Path | str,
    guide: str | None = None,
    holdout_frac: float = 0.1,
) -> dict[str, int]:
    """Turn `chosen` replies into a `--data` directory `train-sft` can consume.

    DPO can only rank what the policy already produces. `chosen` is off
    distribution — never in the pretraining corpus (the tokenizer has no merge
    for `嘻嘻`, which costs four byte-level tokens) and never in SFT's data — so
    at step 0 only 22% of pairs prefer it, and DPO's cheapest way to satisfy
    that gap is one fixed opening for every prompt. A short SFT pass on these
    replies puts the style in the model first; DPO then only ranks.

    Rows the outline leaked into are dropped when `guide` is given. The
    tokenizer is copied in so the directory works as `--data` unchanged.
    """
    rows = _style_rows(Path(prefs_jsonl), guide)
    if not rows:
        raise ValueError(f"no usable chosen replies in {prefs_jsonl}")
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(tok_dir) / "tokenizer.json", dest / "tokenizer.json")
    cut = max(1, int(len(rows) * (1 - holdout_frac)))
    for name, part in (("sft_train.jsonl", rows[:cut]), ("sft_holdout.jsonl", rows[cut:])):
        with (dest / name).open("w", encoding="utf-8") as handle:
            for row in part:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"train": cut, "holdout": len(rows) - cut}


def _style_rows(prefs_jsonl: Path, guide: str | None) -> list[dict]:
    """`chosen` sides as SFT conversations, in file order."""
    rows: list[dict] = []
    for line in prefs_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        chosen = str(pair.get("chosen", "")).strip()
        if not chosen or is_rejected(chosen):
            continue
        if guide is not None and copies_guide(chosen, guide):
            continue
        rows.append(
            {
                "conversations": [
                    {"role": "user", "content": pair["prompt"]},
                    {"role": "assistant", "content": chosen},
                ]
            }
        )
    return rows


def _skipped_path(dest: Path) -> Path:
    """`prefs.jsonl` -> `prefs.skipped.jsonl`, beside the pairs it explains."""
    return dest.with_suffix(".skipped.jsonl")


def _now() -> str:
    """One `created_at` call site, so both record kinds stamp the same way."""
    return datetime.now(UTC).isoformat()


def _skip_record(prompt: str, reason: str, model: str) -> dict[str, str]:
    """A dropped prompt, kept so the next run does not pay for it again."""
    return {
        "prompt": prompt,
        "reason": reason,
        "model": model,
        "created_at": _now(),
    }


def _append(path: Path, record: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _client(model: str):
    from langchain_deepseek import ChatDeepSeek

    # A request timeout is what turns a hung socket into an exception `_invoke`
    # can retry. Without it two of six shards sat for 12 minutes on one call
    # while their files stopped growing. `max_retries=0` keeps the retry policy
    # in `_invoke`, which classifies transient failures itself.
    return ChatDeepSeek(model=model, temperature=0.8, timeout=120, max_retries=0)
