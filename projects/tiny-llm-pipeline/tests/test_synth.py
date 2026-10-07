"""Synthesize DPO preference pairs with DeepSeek, without touching the network in tests."""

import json
from pathlib import Path

from tiny_llm_pipeline.config import DEFAULT_MODEL
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
    skipped = json.loads((tmp_path / "prefs.skipped.jsonl").read_text(encoding="utf-8").strip())
    assert skipped["prompt"] == "a" and skipped["reason"] == "over_limit"


def test_dropped_prompts_are_not_retried(tmp_path, tok) -> None:
    """A refusal is paid for once: the sidecar keeps it out of the next run."""

    class RefusingLLM:
        def __init__(self) -> None:
            self.calls: list = []

        def invoke(self, msgs):
            self.calls.append(msgs)
            return type("M", (), {"content": "抱歉，我不能帮你。"})

    llm = RefusingLLM()
    p = tmp_path / "prefs.jsonl"
    synth_prefs(["a"], p, n=1, tok=tok, llm=llm)
    assert len(llm.calls) == 2  # 1 prompt x 2 calls, both rejected
    synth_prefs(["a", "b"], p, n=2, tok=tok, llm=llm)
    assert len(llm.calls) == 4  # only "b" is new
    assert not p.exists()       # nothing usable was produced


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
    assert "160 字" in chosen_sys        # a concrete length target
    # The token figure is the cap `_fits` actually enforces, not a stale copy.
    assert f"{DEFAULT_MODEL.max_seq_len} token" in chosen_sys
