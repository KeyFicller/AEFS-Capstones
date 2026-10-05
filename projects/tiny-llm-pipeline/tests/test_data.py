import json

import numpy as np

from tiny_llm_pipeline.data import collate_sft, prepare_pretrain, prepare_sft


def test_uint16_roundtrip(tmp_path, tok, tiny_pretrain_jsonl) -> None:
    train_bin, _val_bin = prepare_pretrain(tiny_pretrain_jsonl, tmp_path, tok, max_tokens=4096)
    arr = np.memmap(train_bin, dtype=np.uint16, mode="r")
    assert arr.dtype == np.uint16 and arr.max() < tok.vocab_size


def test_pretrain_strips_chat_markers(tmp_path, tok) -> None:
    body = "今天天气真好，我们出去玩吧。" * 4
    other = "明天一起去公园里慢慢散步。" * 4
    raw = f"<|im_start|>{body}<|im_end|> <|im_start|>短<|im_end|> <|im_start|>{other}<|im_end|>"
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"text": raw}, ensure_ascii=False) + "\n", encoding="utf-8")
    train_bin, _val_bin = prepare_pretrain(source, tmp_path, tok, max_tokens=4096)
    arr = np.memmap(train_bin, dtype=np.uint16, mode="r")
    decoded = tok.decode(arr.tolist())
    assert "<|im_start|>" not in decoded and "<|im_end|>" not in decoded
    assert decoded == body + other
    assert arr.tolist().count(tok.encode("<|eos|>")[0]) == 2


def test_three_splits_disjoint(tiny_sft_jsonl, tok) -> None:
    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=2, dpo_prompt_n=3)
    train_qs = {
        message["content"].strip()
        for sample in bundle.train
        for message in sample["conversations"]
        if message["role"] == "user"
    }
    hold_qs = {
        message["content"].strip()
        for sample in bundle.holdout
        for message in sample["conversations"]
        if message["role"] == "user"
    }
    dpo_qs = {prompt.strip() for prompt in bundle.dpo_prompts}
    assert not (train_qs & hold_qs) and not (train_qs & dpo_qs) and not (hold_qs & dpo_qs)


def test_sft_collate_shapes_and_mask(tiny_sft_jsonl, tok) -> None:
    bundle = prepare_sft(tiny_sft_jsonl, tok, holdout_n=1, dpo_prompt_n=1)
    batch = collate_sft(bundle.train[:4], tok, max_seq_len=128)
    assert batch["input_ids"].shape == batch["loss_mask"].shape
    assert batch["loss_mask"].sum() > 0
    assert (batch["labels"][~batch["loss_mask"]] == -100).all()
