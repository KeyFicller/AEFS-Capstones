"""SFT loop: response-only loss, and a held-out perplexity that drops."""

import json

import pytest
import torch

from tiny_llm_pipeline.data import prepare_sft
from tiny_llm_pipeline.train import sft as sft_mod
from tiny_llm_pipeline.train.sft import eval_sft_ppl, train_sft


def _use_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sft_mod, "_device", lambda: torch.device("cpu"))


def test_sft_beats_base_on_holdout(tmp_path, tok, tiny_sft_jsonl, make_ckpt, monkeypatch) -> None:
    _use_cpu(monkeypatch)
    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=2, dpo_prompt_n=2)
    base = make_ckpt("base")
    dev = "cpu"
    before = eval_sft_ppl(base, bundle.holdout, tok, dev)
    ckpt = train_sft(
        bundle,
        tmp_path / "sft",
        tok=tok,
        base_ckpt=base,
        tok_hash=tok.hash,
        epochs=1,
        max_steps=20,
        seed=0,
    )
    after = eval_sft_ppl(ckpt, bundle.holdout, tok, dev)
    assert after < before


def test_eval_sft_ppl_empty_holdout_is_inf(tmp_path, tok, make_ckpt) -> None:
    base = make_ckpt("base")
    assert eval_sft_ppl(base, [], tok, "cpu") == float("inf")


def test_step_counter_starts_fresh_from_base(tmp_path, tok, tiny_sft_jsonl, monkeypatch) -> None:
    """A pretrain checkpoint's step must not leak into SFT's counter.

    Carrying it over makes `max_steps` absolute, so a base saved at step 99
    would run zero of the requested steps.
    """
    _use_cpu(monkeypatch)
    from tiny_llm_pipeline.config import DEFAULT_MODEL
    from tiny_llm_pipeline.model import TinyLM, save_ckpt

    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=2, dpo_prompt_n=2)
    base = tmp_path / "base" / "ckpt.pt"
    save_ckpt(base, TinyLM(DEFAULT_MODEL), DEFAULT_MODEL, step=99, tokenizer_hash=tok.hash)
    out = tmp_path / "sft"

    train_sft(bundle, out, tok=tok, base_ckpt=base, tok_hash=tok.hash, max_steps=3)

    records = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [1, 2, 3]
