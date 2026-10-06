"""SFT loop: response-only loss, and a held-out perplexity that drops."""

import json
import logging
from pathlib import Path

import pytest
import torch

from tiny_llm_pipeline.config import DEFAULT_MODEL, DEFAULT_TRAIN
from tiny_llm_pipeline.data import prepare_sft
from tiny_llm_pipeline.model import TinyLM, load_ckpt
from tiny_llm_pipeline.train import sft as sft_mod
from tiny_llm_pipeline.train.sft import _holdout_sample, _log_sft_samples, eval_sft_ppl, train_sft


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


def _log_rows(path: Path) -> list[dict[str, float | int]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_val_every_rejects_zero(tmp_path, tok, tiny_sft_jsonl, make_ckpt) -> None:
    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=2, dpo_prompt_n=2)
    with pytest.raises(ValueError, match="val_every"):
        train_sft(
            bundle,
            tmp_path / "sft",
            tok=tok,
            base_ckpt=make_ckpt("base"),
            tok_hash=tok.hash,
            max_steps=1,
            val_every=0,
        )


def test_ckpt_records_holdout_ppl_and_monitor(
    tmp_path, tok, tiny_sft_jsonl, make_ckpt, monkeypatch
) -> None:
    _use_cpu(monkeypatch)
    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=2, dpo_prompt_n=2)
    out = tmp_path / "sft"
    returned = train_sft(
        bundle,
        out,
        tok=tok,
        base_ckpt=make_ckpt("base"),
        tok_hash=tok.hash,
        max_steps=2,
        ckpt_every=2,
        plot_every=2,
        val_every=-1,
    )
    rows = _log_rows(out / "train_log.jsonl")
    assert [row["step"] for row in rows] == [1, 2]
    assert "val_ppl" not in rows[0]
    assert rows[1]["val_ppl"] > 1
    assert (out / "monitor.png").stat().st_size > 0
    payload = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == 2
    assert payload["cursor"] == 2 * DEFAULT_TRAIN.batch_size
    assert "optimizer" in payload
    assert returned == out / "best.pt"


def test_best_ckpt_keeps_the_lowest_holdout_ppl(
    tmp_path, tok, tiny_sft_jsonl, make_ckpt, monkeypatch
) -> None:
    """A later, worse step must not overwrite `best.pt`, including across a resume."""
    _use_cpu(monkeypatch)
    values = iter([9.0, 4.0, 7.0])
    monkeypatch.setattr(sft_mod, "_response_ppl", lambda *_a, **_k: next(values))
    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=2, dpo_prompt_n=2)
    out = tmp_path / "sft"
    common = {
        "tok": tok,
        "base_ckpt": make_ckpt("base"),
        "tok_hash": tok.hash,
        "ckpt_every": 1,
        "plot_every": 0,
        "val_every": -1,
    }
    train_sft(bundle, out, max_steps=3, **common)
    assert torch.load(out / "best.pt", map_location="cpu", weights_only=False)["step"] == 2
    assert torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)["step"] == 3

    monkeypatch.setattr(sft_mod, "_response_ppl", lambda *_a, **_k: 5.0)
    returned = train_sft(bundle, out, max_steps=4, resume=out / "ckpt.pt", **common)
    assert torch.load(out / "best.pt", map_location="cpu", weights_only=False)["step"] == 2
    assert torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)["step"] == 4
    assert returned == out / "best.pt"


def test_resume_matches_uninterrupted_run(tmp_path, tok, tiny_sft_jsonl, make_ckpt, monkeypatch) -> None:
    _use_cpu(monkeypatch)
    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=2, dpo_prompt_n=2)
    base = make_ckpt("base")
    full = tmp_path / "full"
    part = tmp_path / "part"
    common = {
        "tok": tok,
        "base_ckpt": base,
        "tok_hash": tok.hash,
        "plot_every": 0,
        "val_every": -1,
    }
    train_sft(bundle, full, max_steps=2, ckpt_every=2, **common)
    train_sft(bundle, part, max_steps=1, ckpt_every=1, **common)
    log_path = part / "train_log.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 99, "loss": 1, "lr": 1, "tokens": 1, "elapsed": 1}) + "\n")
    train_sft(bundle, part, max_steps=2, ckpt_every=2, resume=part / "ckpt.pt", **common)
    full_rows = _log_rows(full / "train_log.jsonl")
    part_rows = _log_rows(log_path)
    assert [row["step"] for row in part_rows] == [1, 2]
    assert [row["loss"] for row in part_rows] == [row["loss"] for row in full_rows]
    assert part_rows[1]["tokens"] == full_rows[1]["tokens"]
    assert part_rows[1]["elapsed"] >= part_rows[0]["elapsed"]
    full_weights = load_ckpt(full / "ckpt.pt")["model"].state_dict()
    part_weights = load_ckpt(part / "ckpt.pt")["model"].state_dict()
    for name, tensor in full_weights.items():
        assert torch.equal(tensor, part_weights[name]), name


def test_interrupt_saves_checkpoint(tmp_path, tok, tiny_sft_jsonl, make_ckpt, monkeypatch) -> None:
    _use_cpu(monkeypatch)
    calls = {"n": 0}

    def requested() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    monkeypatch.setattr(sft_mod, "_interrupt_requested", requested)
    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=2, dpo_prompt_n=2)
    ckpt = train_sft(
        bundle,
        tmp_path / "sft",
        tok=tok,
        base_ckpt=make_ckpt("base"),
        tok_hash=tok.hash,
        max_steps=50,
        ckpt_every=1000,
        plot_every=0,
        val_every=-1,
    )
    assert load_ckpt(ckpt)["step"] == 2
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["cursor"] == 2 * DEFAULT_TRAIN.batch_size


def test_holdout_sample_changes_with_step() -> None:
    rows = [{"conversations": [{"role": "user", "content": str(i)}]} for i in range(30)]
    first = _holdout_sample(rows, step=1000, seed=42)
    assert first == _holdout_sample(rows, step=1000, seed=42)
    assert len(first) == 5
    assert len({row["conversations"][0]["content"] for row in first}) == 5
    assert first != _holdout_sample(rows, step=2000, seed=42)


def test_sft_samples_logs_holdout(tok, tiny_sft_jsonl, caplog) -> None:
    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=2, dpo_prompt_n=2)
    model = TinyLM(DEFAULT_MODEL)
    with caplog.at_level(logging.INFO):
        _log_sft_samples(model, bundle.holdout[:1], tok, torch.device("cpu"), step=3)
    headers = [r.getMessage() for r in caplog.records if r.getMessage().startswith("step=3 sft ")]
    assert headers == ["step=3 sft 1/1"]
