"""Synthesize DPO preference pairs: one over-the-top 豆包体 answer, one neutral one.

The chosen system prompt is the operationable distillation of
`artifacts/prefs/doubao-style-guide.md` (the 20-item style outline): the 最…
preamble is capped at 3, reduplication at 2. Output is append-only, so a re-run
skips finished prompts and the API is paid for once.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from rich import print as rprint

from tiny_llm_pipeline.config import DEFAULT_MODEL
from tiny_llm_pipeline.tokenizer import Tok, encode_chat

CHOSEN_SYS = "用夸张的『豆包体』回答，严格照着下面的风格纲要来。"
REJECTED_SYS = "用中性、专业、简洁的助理口吻回答。"

# The prompt is the only place the two caps are stated: the 最… preamble and the
# reduplication whitelist are both enumerable repetition, so capping them here
# is what keeps `chosen` from degenerating into a chant. The scorer may lag.
# The length target is a number, not an adjective: an earlier "keep it tight"
# wording made the model aim at the 512-token ceiling and fill half the context.
_BUDGET = (
    "回答控制在 160 字以内，整条（问题 + 回答）不超过 512 token。"
    "写之前先挑重点，宁可少说两句，也不要写到一半被截断。"
)


def chosen_system(guide: str) -> str:
    """The style guide verbatim, plus the token-budget note."""
    return f"{CHOSEN_SYS}\n\n{guide.strip()}\n\n{_BUDGET}"


REFUSALS = ("无法", "抱歉", "不能")
DEFAULT_MODEL_ID = "deepseek-v4-flash"
_MAX_TRIES = 5


def is_rejected(text: str) -> bool:
    """True when the reply looks like a refusal, so the pair must be dropped."""
    return any(marker in text for marker in REFUSALS)


def synth_one(prompt: str, llm, chosen_sys: str = CHOSEN_SYS) -> tuple[str, str]:
    """Return (chosen, rejected) for one prompt."""
    return _invoke(llm, chosen_sys, prompt), _invoke(llm, REJECTED_SYS, prompt)


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
    `sleep` throttles between prompts; retries back off on their own.
    """
    dest = Path(out_jsonl)
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_prompts(dest) if resume else set()
    chosen_sys = chosen_system(guide) if guide else CHOSEN_SYS
    client = llm if llm is not None else _client(model)
    dropped = 0
    over_limit = 0
    for prompt in prompts[:n]:
        if prompt in done:
            continue
        chosen, rejected = synth_one(prompt, client, chosen_sys)
        if _drop(chosen, rejected):
            dropped += 1
            rprint(f"[yellow]dropped[/yellow] {dropped} total: {prompt[:40]}")
            continue
        if not _fits(prompt, chosen, rejected, tok, max_len):
            over_limit += 1
            rprint(f"[yellow]over {max_len} tokens[/yellow] ({over_limit}): {prompt[:40]}")
            continue
        _append(
            dest,
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "model": model,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        done.add(prompt)
        if sleep:
            time.sleep(sleep)
    if over_limit:
        rprint(f"[yellow]{over_limit} pairs dropped for exceeding {max_len} tokens[/yellow]")
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


def _drop(chosen: str, rejected: str) -> bool:
    if not chosen or not rejected or chosen == rejected:
        return True
    return is_rejected(chosen) or is_rejected(rejected)


def _invoke(llm, system: str, prompt: str) -> str:
    """Call the model, retrying only transient failures.

    Messages are `(role, content)` tuples, not `SystemMessage`, so tests can
    read the system prompt as `msgs[0][1]`.
    """
    messages = [("system", system), ("user", prompt)]
    delay = 1.0
    for attempt in range(_MAX_TRIES):
        try:
            return llm.invoke(messages).content.strip()
        except Exception as exc:
            if attempt == _MAX_TRIES - 1 or not _transient(exc):
                raise
            time.sleep(delay)
            delay *= 2
    raise AssertionError("retry loop exhausted")


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


def _append(path: Path, record: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _client(model: str):
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(model=model, temperature=0.8)
