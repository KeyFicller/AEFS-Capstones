import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from tiny_llm_pipeline.config import DEFAULT_MODEL, DEFAULT_TRAIN
from tiny_llm_pipeline.data import prepare_pretrain
from tiny_llm_pipeline.model import TinyLM, load_ckpt
from tiny_llm_pipeline.train import pretrain as pretrain_mod
from tiny_llm_pipeline.train.pretrain import (
    _boundary_targets,
    _log_val_samples,
    _on_sigint,
    _sample_span,
    train_pretrain,
)


def test_pretrain_runs_and_writes_ckpt(tmp_path, tok, tiny_pretrain_jsonl) -> None:
    prepare_pretrain(tiny_pretrain_jsonl, tmp_path, tok, max_tokens=4096)
    ckpt = train_pretrain(
        tmp_path,
        tmp_path / "out",
        tok_hash=tok.hash,
        max_steps=20,
        max_tokens=2_000_000,
        ckpt_every=20,
    )
    loaded = load_ckpt(ckpt, expect_tokenizer_hash=tok.hash)
    assert loaded["tokenizer_hash"] == tok.hash and loaded["step"] == 20


def test_ckpt_rejects_tokenizer_mismatch(tmp_path, tok, tiny_pretrain_jsonl) -> None:
    prepare_pretrain(tiny_pretrain_jsonl, tmp_path, tok, max_tokens=1024)
    ckpt = train_pretrain(
        tmp_path,
        tmp_path / "out",
        tok_hash=tok.hash,
        max_steps=2,
        max_tokens=100_000,
        ckpt_every=2,
    )
    with pytest.raises(ValueError):
        load_ckpt(ckpt, expect_tokenizer_hash="deadbeef")


def test_val_every_rejects_zero() -> None:
    with pytest.raises(ValueError, match="val_every"):
        train_pretrain(
            Path("missing"),
            Path("out"),
            tok_hash="ab",
            max_steps=1,
            max_tokens=1,
            val_every=0,
        )


def test_token_after_eos_is_ignored() -> None:
    x = torch.tensor([[4, 2, 7]])
    y = torch.tensor([[2, 7, 8]])
    assert _boundary_targets(x, y).tolist() == [[2, -100, 8]]
    assert y.tolist() == [[2, 7, 8]]


def test_val_sample_keeps_a_continuation() -> None:
    prefix, gold = _sample_span(24)
    assert prefix == 12 and gold == 12
    prefix, gold = _sample_span(80)
    assert prefix == 32 and gold == 32


def test_val_samples_logs_five(tmp_path, tok, caplog) -> None:
    body = np.arange(80, dtype=np.uint16) % 40 + 10
    piece = np.concatenate([body, np.array([2], dtype=np.uint16)])
    ids = np.tile(piece, 5)
    val_bin = tmp_path / "val.bin"
    ids.tofile(val_bin)
    model = TinyLM(DEFAULT_MODEL)
    with caplog.at_level(logging.INFO):
        _log_val_samples(model, val_bin, tok, torch.device("cpu"), step=3)
    headers = [r.getMessage() for r in caplog.records if r.getMessage().startswith("step=3 val ")]
    assert headers == [f"step=3 val {i}/5" for i in range(1, 6)]


def _use_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pretrain_mod, "_device", lambda: torch.device("cpu"))


def test_device_prefers_cuda_then_mps_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pretrain_mod._device().type == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pretrain_mod._device().type == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pretrain_mod._device().type == "cpu"


def _log_rows(path: Path) -> list[dict[str, float | int]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_resume_matches_uninterrupted_run(tmp_path, tok, tiny_pretrain_jsonl, monkeypatch) -> None:
    _use_cpu(monkeypatch)
    prepare_pretrain(tiny_pretrain_jsonl, tmp_path, tok, max_tokens=4096)
    full = tmp_path / "full"
    part = tmp_path / "part"
    train_pretrain(
        tmp_path, full, tok_hash=tok.hash, max_steps=2, max_tokens=2_000_000, ckpt_every=2
    )
    train_pretrain(
        tmp_path, part, tok_hash=tok.hash, max_steps=1, max_tokens=2_000_000, ckpt_every=1
    )
    log_path = part / "train_log.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 99, "loss": 1, "lr": 1, "tokens": 1, "elapsed": 1}) + "\n")
    train_pretrain(
        tmp_path,
        part,
        tok_hash=tok.hash,
        max_steps=2,
        max_tokens=2_000_000,
        ckpt_every=2,
        resume=part / "ckpt.pt",
    )
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


def test_resume_old_ckpt_keeps_the_data_cursor(
    tmp_path, tok, tiny_pretrain_jsonl, monkeypatch
) -> None:
    _use_cpu(monkeypatch)
    prepare_pretrain(tiny_pretrain_jsonl, tmp_path, tok, max_tokens=4096)
    out = tmp_path / "out"
    train_pretrain(
        tmp_path, out, tok_hash=tok.hash, max_steps=1, max_tokens=2_000_000, ckpt_every=1
    )
    ckpt = out / "ckpt.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    del payload["tokens"]
    del payload["cursor"]
    torch.save(payload, ckpt)
    train_pretrain(
        tmp_path,
        out,
        tok_hash=tok.hash,
        max_steps=2,
        max_tokens=2_000_000,
        ckpt_every=2,
        resume=ckpt,
    )
    restored = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert restored["step"] == 2
    assert restored["cursor"] == 2 * DEFAULT_TRAIN.batch_size
    assert [row["step"] for row in _log_rows(out / "train_log.jsonl")] == [1, 2]


def test_second_sigint_aborts() -> None:
    pretrain_mod._INTERRUPT = False
    _on_sigint(2, None)
    assert pretrain_mod._INTERRUPT
    with pytest.raises(KeyboardInterrupt):
        _on_sigint(2, None)
    pretrain_mod._INTERRUPT = False


def test_interrupt_saves_checkpoint(tmp_path, tok, tiny_pretrain_jsonl, monkeypatch) -> None:
    _use_cpu(monkeypatch)
    calls = {"n": 0}

    def requested() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    monkeypatch.setattr(pretrain_mod, "_interrupt_requested", requested)
    prepare_pretrain(tiny_pretrain_jsonl, tmp_path, tok, max_tokens=4096)
    ckpt = train_pretrain(
        tmp_path,
        tmp_path / "out",
        tok_hash=tok.hash,
        max_steps=50,
        max_tokens=2_000_000,
        ckpt_every=1000,
    )
    assert load_ckpt(ckpt)["step"] == 2
    assert torch.load(ckpt, map_location="cpu", weights_only=False)["cursor"] == (
        2 * DEFAULT_TRAIN.batch_size
    )
