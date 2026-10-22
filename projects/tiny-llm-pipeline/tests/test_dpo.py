"""DPO: the loss against a manual derivation, the masked log-prob, the frozen ref."""

import json
import math
from pathlib import Path

import pytest
import torch

from tiny_llm_pipeline.data import collate_dpo
from tiny_llm_pipeline.model import load_ckpt
from tiny_llm_pipeline.train.dpo import _is_holdout, dpo_loss, seq_logprob, train_dpo


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


def test_resume_continues_the_step_counter(
    tmp_path, tok, make_ckpt, tiny_prefs_jsonl, interrupt_after, monkeypatch
) -> None:
    """Resuming an existing run reloads AdamW state instead of starting fresh."""
    from tiny_llm_pipeline.train import dpo as dpo_mod

    ref = make_ckpt("sft")
    out = tmp_path / "dpo"
    interrupt_after(dpo_mod, 1)
    train_dpo(tiny_prefs_jsonl, out, ref_ckpt=ref, tok_hash=tok.hash, tok=tok, max_steps=3)
    monkeypatch.setattr(dpo_mod, "_interrupt_requested", lambda: False)
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


def test_second_fresh_run_replaces_the_log(tmp_path, tok, make_ckpt, tiny_prefs_jsonl) -> None:
    """The delivered DPO log held two runs: 208 rows for 104 steps."""
    ref = make_ckpt("sft")
    out = tmp_path / "dpo"
    for _ in range(2):
        train_dpo(tiny_prefs_jsonl, out, ref_ckpt=ref, tok_hash=tok.hash, tok=tok, max_steps=1)
    records = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [1]


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


def test_holdout_split_is_stable_and_keyed_by_prompt() -> None:
    prompts = [f"问题{i}？" for i in range(2000)]
    held = {prompt for prompt in prompts if _is_holdout(prompt, 0.1)}
    assert 150 <= len(held) <= 250
    assert held == {prompt for prompt in prompts if _is_holdout(prompt, 0.1)}
    assert not any(_is_holdout(prompt, 0.0) for prompt in prompts)


def test_holdout_readings_are_logged(tmp_path, tok, make_ckpt) -> None:
    prefs = tmp_path / "prefs.jsonl"
    rows = [
        {"prompt": f"问题{i}？", "chosen": f"宝子～回答{i}呢！", "rejected": f"回答{i}。"}
        for i in range(200)
    ]
    prefs.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    out = tmp_path / "dpo"
    train_dpo(
        prefs,
        out,
        ref_ckpt=make_ckpt("sft"),
        tok_hash=tok.hash,
        tok=tok,
        max_steps=1,
        ckpt_every=1,
    )
    records = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    record = records[0]
    assert "holdout_gap" in record and "holdout_margin" in record
    assert 0.0 <= record["holdout_win_rate"] <= 1.0
    # The sum form favours the shorter side; the per-token form does not, and
    # the two disagreeing is what exposed the length confound in the first run.
    assert 0.0 <= record["holdout_win_rate_norm"] <= 1.0
    assert "holdout_gap_norm" in record


def test_length_normalized_readings_separate_style_from_length(
    tmp_path, tok, make_ckpt
) -> None:
    """A pair whose sides differ only in length wins on sum and loses per token."""
    from tiny_llm_pipeline.data import read_preferences
    from tiny_llm_pipeline.model import load_ckpt
    from tiny_llm_pipeline.train.dpo import _holdout_readings

    prefs = tmp_path / "prefs.jsonl"
    prefs.write_text(
        json.dumps(
            {"prompt": "问题？", "chosen": "好" * 60, "rejected": "好"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = read_preferences(prefs)
    model = load_ckpt(make_ckpt("sft"))["model"]
    readings = _holdout_readings(model, model, rows, tok, 768, 0.1, torch.device("cpu"))
    assert readings is not None
    # Same policy on both sides, so the β-scaled margin is exactly zero.
    assert readings["holdout_margin"] == pytest.approx(0.0, abs=1e-6)
    assert readings["holdout_gap"] != readings["holdout_gap_norm"]


def test_guard_trips_on_collapsed_accuracy() -> None:
    """The delivered failure: 9.5% accuracy against a 37.2% reference."""
    from tiny_llm_pipeline.train.dpo import _guard_trips

    assert _guard_trips(0.095, 0.372, chosen_up=32.2, rejected_down=183.3)
    # The controlled 400-step run stayed healthy on both readings.
    assert _guard_trips(0.337, 0.372, chosen_up=81.2, rejected_down=65.2) is None
    # A margin bought from below is rejected even when accuracy holds.
    assert _guard_trips(0.37, 0.372, chosen_up=5.0, rejected_down=90.0)
    # No probe file, no accuracy rule.
    assert _guard_trips(None, None, chosen_up=40.0, rejected_down=10.0) is None


def test_guard_is_logged_when_a_probe_is_passed(tmp_path, tok, make_ckpt) -> None:
    probe = [
        {"conversations": [
            {"role": "user", "content": "问题？"},
            {"role": "assistant", "content": "好的。"},
        ]}
    ]
    # 20 pairs so a 50% hash split is certain to leave a holdout to read the
    # chosen/rejected split from.
    prefs = tmp_path / "prefs.jsonl"
    prefs.write_text(
        "\n".join(
            json.dumps({"prompt": f"问题{i}？", "chosen": f"宝子～回答{i}呢！", "rejected": f"回答{i}。"},
                       ensure_ascii=False)
            for i in range(20)
        ),
        encoding="utf-8",
    )
    out = tmp_path / "dpo"
    train_dpo(
        prefs,
        out,
        ref_ckpt=make_ckpt("sft"),
        tok_hash=tok.hash,
        tok=tok,
        max_steps=10,
        ckpt_every=10,
        holdout_frac=0.5,
        probe=probe,
    )
    records = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert "guard_acc" in records[-1]
    assert "guard_chosen_up" in records[-1]
    assert "guard_rejected_down" in records[-1]
