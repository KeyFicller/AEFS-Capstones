"""LoRA: identity at init, merge equivalence, parameter count, adapter I/O."""

import json
from dataclasses import replace

import pytest
import torch

from tiny_llm_pipeline.config import DEFAULT_MODEL, DEFAULT_TRAIN, SFT_LR
from tiny_llm_pipeline.data import prepare_sft
from tiny_llm_pipeline.lora import (
    LoRAConfig,
    LoRALinear,
    adapter_state_dict,
    inject_lora,
    load_adapter_state,
    merge_lora,
)
from tiny_llm_pipeline.model import (
    TinyLM,
    load_ckpt,
    load_lora_ckpt,
    save_ckpt,
    save_lora_ckpt,
)
from tiny_llm_pipeline.train import sft as sft_mod
from tiny_llm_pipeline.train.sft import eval_sft_ppl, train_sft

_LORA = LoRAConfig()
_TRAINABLE = 212_992


def _model(seed: int = 0) -> TinyLM:
    torch.manual_seed(seed)
    return TinyLM(DEFAULT_MODEL)


def _batch() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, DEFAULT_MODEL.vocab_size, (2, 8))


def _randomize(model: TinyLM) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.a.normal_(0, 0.05)
                module.b.normal_(0, 0.05)


def test_injection_is_identity_at_init_and_counts_parameters() -> None:
    model = _model()
    idx = _batch()
    before = model(idx)[0]

    assert inject_lora(model, _LORA) == _TRAINABLE
    assert torch.equal(before, model(idx)[0])

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(name.endswith((".a", ".b")) for name in trainable)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == _TRAINABLE


def test_merge_matches_the_wrapper() -> None:
    model = _model()
    inject_lora(model, _LORA)
    _randomize(model)
    idx = _batch()
    wrapped = model(idx)[0]

    merged = merge_lora(model)

    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    assert torch.allclose(wrapped, merged(idx)[0], atol=1e-4)


def test_adapter_state_round_trips_onto_the_base() -> None:
    source = _model()
    inject_lora(source, _LORA)
    _randomize(source)
    idx = _batch()
    expected = source(idx)[0]

    other = _model()  # same seed -> same base weights, independent tensors
    inject_lora(other, _LORA)
    load_adapter_state(other, adapter_state_dict(source))

    assert torch.allclose(expected, other(idx)[0], atol=1e-6)


def test_load_adapter_state_rejects_mismatched_layers() -> None:
    model = _model()
    inject_lora(model, _LORA)
    with pytest.raises(ValueError, match="do not match"):
        load_adapter_state(model, {})


def test_zero_rank_is_a_no_op() -> None:
    model = _model()
    before = {name: p.clone() for name, p in model.named_parameters()}
    assert inject_lora(model, LoRAConfig(rank=0)) == 0
    assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())


def _base_and_adapter(tmp_path, tok):
    base = tmp_path / "base" / "ckpt.pt"
    save_ckpt(base, _model(), DEFAULT_MODEL, step=0, tokenizer_hash=tok.hash)
    injected = _model()
    inject_lora(injected, _LORA)
    _randomize(injected)
    adapter = tmp_path / "adapter" / "ckpt.pt"
    save_lora_ckpt(adapter, injected, DEFAULT_MODEL, _LORA, base, step=3, tokenizer_hash=tok.hash)
    return base, injected, adapter


def test_lora_ckpt_round_trip(tmp_path, tok) -> None:
    _, injected, adapter = _base_and_adapter(tmp_path, tok)
    idx = _batch()

    payload = torch.load(adapter, map_location="cpu", weights_only=False)
    assert "lora" in payload and "model" not in payload
    assert payload["base"].endswith("base/ckpt.pt")

    merged = load_ckpt(adapter)["model"]
    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    assert torch.allclose(injected(idx)[0], merged(idx)[0], atol=1e-4)
    assert load_ckpt(adapter)["step"] == 3

    injected_again, lora_cfg = load_lora_ckpt(adapter)
    assert lora_cfg == _LORA
    assert torch.allclose(injected(idx)[0], injected_again(idx)[0], atol=1e-6)


def test_lora_ckpt_error_paths(tmp_path, tok) -> None:
    base, injected, adapter = _base_and_adapter(tmp_path, tok)

    with pytest.raises(ValueError, match="tokenizer hash"):
        load_ckpt(adapter, expect_tokenizer_hash="nope")

    nested = tmp_path / "nested" / "ckpt.pt"
    save_lora_ckpt(nested, injected, DEFAULT_MODEL, _LORA, adapter, step=0, tokenizer_hash=tok.hash)
    with pytest.raises(ValueError, match="itself an adapter"):
        load_ckpt(nested)

    payload = torch.load(adapter, map_location="cpu", weights_only=False)
    payload.pop("base")
    stripped = tmp_path / "stripped" / "ckpt.pt"
    stripped.parent.mkdir(parents=True)
    torch.save(payload, stripped)
    with pytest.raises(ValueError, match="no base"):
        load_ckpt(stripped)

    payload = torch.load(adapter, map_location="cpu", weights_only=False)
    payload["cfg"] = {**payload["cfg"], "d_model": 256}
    bad = tmp_path / "bad" / "ckpt.pt"
    bad.parent.mkdir(parents=True)
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="config"):
        load_ckpt(bad)


def test_lora_ckpt_base_can_be_overridden(tmp_path, tok) -> None:
    base, _, adapter = _base_and_adapter(tmp_path, tok)
    moved = tmp_path / "moved" / "ckpt.pt"
    moved.parent.mkdir(parents=True)
    moved.write_bytes(base.read_bytes())
    assert load_ckpt(adapter, base=moved)["step"] == 3


def _use_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sft_mod, "_device", lambda: torch.device("cpu"))


def test_lora_sft_beats_base_and_writes_an_adapter(
    tmp_path, tok, tiny_sft_jsonl, make_ckpt, monkeypatch
) -> None:
    _use_cpu(monkeypatch)
    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=2, dpo_prompt_n=2)
    base = make_ckpt("base")
    before = eval_sft_ppl(base, bundle.holdout, tok, "cpu")

    ckpt = train_sft(
        bundle,
        tmp_path / "sft-lora",
        tok=tok,
        base_ckpt=base,
        tok_hash=tok.hash,
        max_steps=20,
        seed=0,
        lora=_LORA,
        train_cfg=replace(DEFAULT_TRAIN, lr=SFT_LR),
        plot_every=0,
        val_every=-1,
    )

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "lora" in payload and "model" not in payload
    assert eval_sft_ppl(ckpt, bundle.holdout, tok, "cpu") < before


def test_lora_resume_continues_the_step_counter(
    tmp_path, tok, tiny_sft_jsonl, make_ckpt, monkeypatch
) -> None:
    _use_cpu(monkeypatch)
    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=2, dpo_prompt_n=2)
    out = tmp_path / "sft-lora"
    common = {
        "tok": tok,
        "base_ckpt": make_ckpt("base"),
        "tok_hash": tok.hash,
        "lora": _LORA,
        "plot_every": 0,
        "val_every": -1,
    }
    train_sft(bundle, out, max_steps=1, ckpt_every=1, **common)
    train_sft(bundle, out, max_steps=2, ckpt_every=2, resume=out / "ckpt.pt", **common)

    rows = [
        json.loads(line)["step"]
        for line in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == [1, 2]
    assert "lora" in torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)


def test_zero_rank_lora_is_plain_full_finetuning(
    tmp_path, tok, tiny_sft_jsonl, make_ckpt, monkeypatch
) -> None:
    _use_cpu(monkeypatch)
    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=2, dpo_prompt_n=2)
    out = tmp_path / "sft"

    ckpt = train_sft(
        bundle,
        out,
        tok=tok,
        base_ckpt=make_ckpt("base"),
        tok_hash=tok.hash,
        max_steps=2,
        lora=LoRAConfig(rank=0),
        plot_every=0,
        val_every=-1,
    )

    assert "model" in torch.load(ckpt, map_location="cpu", weights_only=False)
