"""Tests for `tiny_llm_pipeline.generate`: seeds, readings, and the stage bundle."""

import torch
from tiny_llm_pipeline.generate import STAGES, Bundle, Generator, GenSettings, seed_for, stats


class _RecordingModel:
    """Appends a fixed tail and records what it was fed."""

    def __init__(self, tail: list[int] | None = None) -> None:
        self.tail = tail or []
        self.fed: list[torch.Tensor] = []
        self.max_new: list[int] = []

    def generate(self, idx, max_new_tokens, **kwargs):
        self.fed.append(idx.clone())
        self.max_new.append(max_new_tokens)
        if not self.tail:
            return idx
        extra = torch.tensor([self.tail], dtype=idx.dtype, device=idx.device)
        return torch.cat((idx, extra), dim=1)


def test_seed_is_stable_and_order_independent() -> None:
    assert seed_for(0, "sft", 1.5, "你好", 0) == seed_for(0, "sft", 1.5, "你好", 0)
    assert seed_for(0, "sft", 1.5, "你好", 0) != seed_for(0, "sft", 1.5, "你好", 1)
    assert seed_for(0, "sft", 1.5, "你好", 0) != seed_for(0, "dpo", 1.5, "你好", 0)
    assert 0 <= seed_for(0, "sft", 1.5, "你好", 0) < 2**32


def test_stats_readings() -> None:
    s = stats("我用最直白地说。")
    assert s["chars"] == 8
    assert s["tempo"] == 1
    assert 0.0 < s["dist4"] <= 1.0
    assert stats("")["dist4"] == 0.0        # no 4-grams, must not divide by zero


def test_pretrain_encodes_bare_and_chat_stages_use_the_template(tok) -> None:
    from tiny_llm_pipeline.tokenizer import encode_chat

    model = _RecordingModel()
    settings = GenSettings(max_new=1, temperature=0)
    cpu = torch.device("cpu")
    Generator("pretrain", model, tok, end_id=5, device=cpu).reply("你好", settings)
    Generator("sft", model, tok, end_id=5, device=cpu).reply("你好", settings)
    bare, chat = model.fed[0][0].tolist(), model.fed[1][0].tolist()
    assert bare == tok.encode("你好")
    expected, _ = encode_chat([{"role": "user", "content": "你好"}], tok, add_assistant=True)
    assert chat == expected


def test_reply_stops_at_the_end_token_and_passes_max_new(tok) -> None:
    end = tok.encode("<|end|>")[0]
    model = _RecordingModel(tail=[11, end, 12, 13])
    gen = Generator("sft", model, tok, end_id=end, device=torch.device("cpu"))
    assert gen.reply("你好", GenSettings(max_new=7, temperature=0)) == tok.decode([11]).strip()
    assert model.max_new == [7]             # max_new is handed to generate unchanged


class _RandomModel:
    """Draws random ids, the way `TinyLM.generate` does under sampling.

    Ids stay above `end_id`, so truncation never kicks in and the whole tail is
    decoded. An untrained model would just emit `<|end|>` first and truncate
    everything away, which hides what this test is about: the derived seed.
    """

    def __init__(self, end_id: int, vocab: int = 500) -> None:
        self.end_id = end_id
        self.vocab = vocab

    def generate(self, idx, max_new_tokens, **kwargs):
        extra = torch.randint(
            self.end_id + 1, self.vocab, (1, max_new_tokens), device=idx.device
        )
        return torch.cat((idx, extra), dim=1)


def test_sampling_is_reproducible(tok) -> None:
    end = tok.encode("<|end|>")[0]
    gen = Generator("sft", _RandomModel(end), tok, end_id=end, device=torch.device("cpu"))
    settings = GenSettings(max_new=16, temperature=0.8)
    assert gen.reply("你好", settings) == gen.reply("你好", settings)
    # index=0 and index=1 derive different seeds, so the draw differs.
    assert gen.reply("你好", settings, index=1) != gen.reply("你好", settings, index=0)


def test_bundle_load_reports_ckpt_steps(tmp_path, tok, make_ckpt) -> None:
    root = tmp_path / "bundle"
    (root / "tokenizer").mkdir(parents=True)
    (root / "tokenizer" / "tokenizer.json").write_bytes(tok.path.read_bytes())
    for stage in STAGES:
        (root / stage).mkdir()
        (root / stage / "ckpt.pt").write_bytes(make_ckpt(stage).read_bytes())
    bundle = Bundle.load(root, device=torch.device("cpu"))
    assert bundle.steps == dict.fromkeys(STAGES, 0)
    replies = bundle.run("你好", GenSettings(max_new=2, temperature=0))
    assert [r.stage for r in replies] == list(STAGES)
