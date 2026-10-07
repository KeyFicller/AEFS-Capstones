"""DPO: the loss against a manual derivation, the masked log-prob, the frozen ref."""

import json
import math
from pathlib import Path

import pytest
import torch

from tiny_llm_pipeline.data import collate_dpo
from tiny_llm_pipeline.model import load_ckpt
from tiny_llm_pipeline.train.dpo import dpo_loss, seq_logprob, train_dpo


def test_dpo_loss_matches_manual() -> None:
    pw, pl, rw, rl, beta = -1.0, -3.0, -1.5, -2.5, 0.1
    loss, margin = dpo_loss(
        torch.tensor(pw), torch.tensor(pl), torch.tensor(rw), torch.tensor(rl), beta
    )
    assert torch.isclose(margin, torch.tensor(beta * ((pw - rw) - (pl - rl))))
    assert torch.isclose(loss, -torch.nn.functional.logsigmoid(margin))


def test_beta_zero_gives_ln2() -> None:
    loss, _ = dpo_loss(
        torch.tensor(-1.0), torch.tensor(-2.0), torch.tensor(-2.0), torch.tensor(-1.0), 0.0
    )
    assert torch.isclose(loss, torch.tensor(math.log(2)))


def test_seq_logprob_skips_unmasked_and_padding(tok) -> None:
    """A gather-based sum must equal a hand-rolled loop over the masked targets."""
    from tiny_llm_pipeline.config import DEFAULT_MODEL
    from tiny_llm_pipeline.model import TinyLM

    torch.manual_seed(0)
    model = TinyLM(DEFAULT_MODEL)
    model.eval()
    batch = collate_dpo(
        [{"prompt": "问题？", "chosen": "宝子～好呀呢！", "rejected": "好的。"}],
        tok,
        DEFAULT_MODEL.max_seq_len,
    )
    ids = batch["chosen_input_ids"]
    mask = batch["chosen_mask"]

    got = seq_logprob(model, ids, mask)
    logits, _ = model(ids[:, :-1])
    logp = torch.log_softmax(logits, dim=-1)
    expected = torch.zeros(ids.size(0))
    for b in range(ids.size(0)):
        for t in range(1, ids.size(1)):
            if mask[b, t]:
                expected[b] += logp[b, t - 1, ids[b, t]]

    assert torch.allclose(got, expected, atol=1e-5)
    # An all-False mask counts nothing, so padding cannot leak into the sum.
    assert torch.allclose(seq_logprob(model, ids, torch.zeros_like(mask)), torch.zeros(1))


def test_ref_weights_frozen_and_policy_moves(tmp_path, tok, make_ckpt, tiny_prefs_jsonl) -> None:
    ref_ck = make_ckpt("sft")
    before = {k: v.clone() for k, v in load_ckpt(ref_ck)["model"].state_dict().items()}
    out = tmp_path / "dpo"

    ckpt = train_dpo(
        tiny_prefs_jsonl, out, ref_ckpt=ref_ck, tok_hash=tok.hash, tok=tok, max_steps=3
    )

    after = load_ckpt(ref_ck)["model"].state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    policy = load_ckpt(ckpt)["model"].state_dict()
    assert any(not torch.equal(before[k], policy[k]) for k in before)
    records = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert len(records) == 3
    assert all("margin" in row and "loss" in row for row in records)


def test_step_counter_starts_fresh_from_ref(tmp_path, tok, tiny_prefs_jsonl) -> None:
    """The SFT checkpoint's step must not leak into DPO's counter."""
    from tiny_llm_pipeline.config import DEFAULT_MODEL
    from tiny_llm_pipeline.model import TinyLM, save_ckpt

    base = tmp_path / "base" / "ckpt.pt"
    save_ckpt(base, TinyLM(DEFAULT_MODEL), DEFAULT_MODEL, step=99, tokenizer_hash=tok.hash)
    out = tmp_path / "dpo"

    train_dpo(tiny_prefs_jsonl, out, ref_ckpt=base, tok_hash=tok.hash, tok=tok, max_steps=3)

    records = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [1, 2, 3]


def test_drops_rows_with_no_completion_tokens(tmp_path, tok, make_ckpt) -> None:
    """A prompt that fills the context leaves no assistant tokens to score."""
    from tiny_llm_pipeline.config import DEFAULT_MODEL

    prefs = tmp_path / "prefs.jsonl"
    long_prompt = "问题" * DEFAULT_MODEL.max_seq_len
    rows = [
        {"prompt": long_prompt, "chosen": "宝子～", "rejected": "好的。"},
        {"prompt": "短问题？", "chosen": "宝子～好呀呢！", "rejected": "好的。"},
    ]
    prefs.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    out = tmp_path / "dpo"
    train_dpo(
        prefs, out, ref_ckpt=make_ckpt("sft"), tok_hash=tok.hash, tok=tok, max_steps=1
    )
    records = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert records[0]["dropped"] == 1  # only the usable row is scored


def test_resume_continues_the_step_counter(tmp_path, tok, make_ckpt, tiny_prefs_jsonl) -> None:
    """Resuming an existing run reloads AdamW state instead of starting fresh."""
    ref = make_ckpt("sft")
    out = tmp_path / "dpo"
    train_dpo(tiny_prefs_jsonl, out, ref_ckpt=ref, tok_hash=tok.hash, tok=tok, max_steps=1)
    train_dpo(
        tiny_prefs_jsonl,
        out,
        ref_ckpt=ref,
        tok_hash=tok.hash,
        tok=tok,
        max_steps=3,
        resume=out / "ckpt.pt",
    )
    records = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [1, 2, 3]
    assert load_ckpt(out / "ckpt.pt")["step"] == 3


def test_resume_rejects_a_checkpoint_without_optimizer_state(
    tmp_path, tok, make_ckpt, tiny_prefs_jsonl
) -> None:
    """A truncated checkpoint must fail loudly, as it does in pretrain and SFT."""
    ref = make_ckpt("sft")
    out = tmp_path / "dpo"
    train_dpo(tiny_prefs_jsonl, out, ref_ckpt=ref, tok_hash=tok.hash, tok=tok, max_steps=1)
    ckpt = out / "ckpt.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    del payload["optimizer"]
    torch.save(payload, ckpt)
    with pytest.raises(RuntimeError, match="no optimizer state"):
        train_dpo(
            tiny_prefs_jsonl,
            out,
            ref_ckpt=ref,
            tok_hash=tok.hash,
            tok=tok,
            max_steps=2,
            resume=ckpt,
        )


def test_interrupt_saves_and_stops(tmp_path, tok, make_ckpt, tiny_prefs_jsonl, monkeypatch) -> None:
    """Ctrl-C finishes the current step and writes a checkpoint, with ckpt_every off."""
    steps = {"n": 0}

    def _interrupt_after_two() -> bool:
        steps["n"] += 1
        return steps["n"] >= 2

    monkeypatch.setattr("tiny_llm_pipeline.train.dpo._interrupt_requested", _interrupt_after_two)
    out = tmp_path / "dpo"
    train_dpo(
        tiny_prefs_jsonl,
        out,
        ref_ckpt=make_ckpt("sft"),
        tok_hash=tok.hash,
        tok=tok,
        max_steps=5,
        ckpt_every=0,
    )
    records = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [1, 2]
    assert load_ckpt(out / "ckpt.pt")["step"] == 2
