from tiny_llm_pipeline.tokenizer import SPECIAL_TOKENS, encode_chat, render_chat


def test_template_exact() -> None:
    msgs = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "在呢"},
    ]
    assert render_chat(msgs, add_assistant=True) == (
        "<|bos|><|user|>你好<|end|><|assistant|>在呢<|end|>"
    )
    assert render_chat(msgs[:1], add_assistant=True) == (
        "<|bos|><|user|>你好<|end|><|assistant|>"
    )


def test_loss_mask_only_on_assistant(tok) -> None:
    ids, mask = encode_chat(
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "在呢"},
        ],
        tok,
    )
    text_on = tok.decode([i for i, keep in zip(ids, mask, strict=True) if keep])
    assert "在呢" in text_on and "你好" not in text_on


def test_special_token_ids_are_fixed(tok) -> None:
    assert [tok.encode(token)[0] for token in SPECIAL_TOKENS] == list(range(len(SPECIAL_TOKENS)))
