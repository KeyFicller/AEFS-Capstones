"""Synthesize DPO preference pairs with DeepSeek, without touching the network in tests."""

import json
from pathlib import Path

from tiny_llm_pipeline.pref.synthesize import is_rejected, synth_prefs


class FakeLLM:
    """Distinguishes the two calls by the 豆包 marker in the system prompt."""

    def __init__(self) -> None:
        self.calls: list = []

    def invoke(self, msgs):
        self.calls.append(msgs)
        content = "好呀宝子～" if "豆包" in msgs[0][1] else "好的。"
        return type("M", (), {"content": content})


def test_filters_and_resumes(tmp_path, tok) -> None:
    llm = FakeLLM()
    p = tmp_path / "prefs.jsonl"
    synth_prefs(["a", "b"], p, n=2, tok=tok, llm=llm)
    assert len(llm.calls) == 4  # 2 prompts x 2 calls
    synth_prefs(["a", "b", "c"], p, n=3, tok=tok, llm=llm)
    assert len(llm.calls) == 6  # only "c" is new
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    assert [row["prompt"] for row in rows] == ["a", "b", "c"]
    assert rows[0]["chosen"] == "好呀宝子～" and rows[0]["rejected"] == "好的。"


def test_drops_pairs_over_the_token_budget(tmp_path, tok) -> None:
    class LongLLM:
        """Both sides are long and different, so only the length guard applies."""

        def invoke(self, msgs):
            text = "宝子" * 400 if "豆包" in msgs[0][1] else "好的" * 400
            return type("M", (), {"content": text})

    p = tmp_path / "prefs.jsonl"
    synth_prefs(["a"], p, n=1, tok=tok, max_len=64, llm=LongLLM())
    assert not p.exists()  # dropped, never truncated


def test_drops_identical_and_refusals() -> None:
    assert is_rejected("抱歉，我不能帮你") is True
    assert is_rejected("好的宝子～") is False


def test_guide_is_embedded_verbatim_in_the_chosen_prompt(tmp_path, tok) -> None:
    """The outline goes in as-is, plus the token budget; nothing is summarized."""
    llm = FakeLLM()
    p = tmp_path / "prefs.jsonl"
    guide = "1. **「最…」排比前摇**——可无限堆叠。\n20. **陪伴与在场感**。"
    synth_prefs(["a"], p, n=1, tok=tok, llm=llm, guide=guide)

    chosen_sys = next(m[0][1] for m in llm.calls if "豆包" in m[0][1])
    assert guide in chosen_sys          # verbatim, not a paraphrase
    assert "512 token" in chosen_sys    # the model is told about the context cap
    assert "160 字" in chosen_sys        # and given a concrete length target
