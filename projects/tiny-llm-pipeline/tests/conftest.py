"""Shared fixtures for the tiny-llm pipeline tests.

`tok` trains a vocab-512 byte-level BPE once per session.
"""

import json

import pytest


@pytest.fixture(scope="session")
def tok(tmp_path_factory):
    from tiny_llm_pipeline.tokenizer import load_tokenizer, train_tokenizer

    d = tmp_path_factory.mktemp("tok")
    text = "你好呀。今天天气真好，我们出去玩吧。\n" * 200
    (d / "sample.txt").write_text(text, encoding="utf-8")
    train_tokenizer(d / "sample.txt", d, vocab_size=512, sample_mb=1)
    return load_tokenizer(d)


@pytest.fixture
def tiny_pretrain_jsonl(tmp_path):
    p = tmp_path / "pre.jsonl"
    rows = [
        json.dumps({"text": "你好呀。今天天气真好，我们出去玩吧。" * 5}, ensure_ascii=False)
        for _ in range(50)
    ]
    p.write_text("\n".join(rows), encoding="utf-8")
    return p


@pytest.fixture
def tiny_sft_jsonl(tmp_path):
    p = tmp_path / "sft.jsonl"
    rows = [
        {
            "conversations": [
                {"role": "user", "content": f"问题{i}？"},
                {"role": "assistant", "content": f"回答{i}。"},
            ]
        }
        for i in range(12)
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return p


@pytest.fixture
def tiny_prefs_jsonl(tmp_path):
    p = tmp_path / "prefs.jsonl"
    rows = [
        {"prompt": f"问题{i}？", "chosen": f"宝子～回答{i}呢！", "rejected": f"回答{i}。"}
        for i in range(8)
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return p


@pytest.fixture
def runner():
    from typer.testing import CliRunner

    return CliRunner()


@pytest.fixture
def make_ckpt(tmp_path, tok):
    from tiny_llm_pipeline.config import DEFAULT_MODEL
    from tiny_llm_pipeline.model import TinyLM, save_ckpt

    def _make(name: str):
        p = tmp_path / name / "ckpt.pt"
        save_ckpt(p, TinyLM(DEFAULT_MODEL), DEFAULT_MODEL, step=0, tokenizer_hash=tok.hash)
        return p

    return _make


@pytest.fixture
def ckpt_paths(make_ckpt):
    return make_ckpt("pretrain"), make_ckpt("sft"), make_ckpt("dpo")
