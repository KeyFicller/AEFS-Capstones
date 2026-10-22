"""Synthesize DPO preference pairs with DeepSeek, without touching the network in tests."""

import json
from pathlib import Path

from tiny_llm_pipeline.config import DEFAULT_MODEL
from tiny_llm_pipeline.pref.synthesize import is_rejected, synth_prefs, write_style_sft


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


def test_rejected_side_rewrites_the_chosen_answer(tmp_path, tok) -> None:
    """The pair must differ in voice only; an independent answer differs in content too."""
    llm = FakeLLM()
    synth_prefs(["a"], tmp_path / "prefs.jsonl", n=1, tok=tok, llm=llm)
    rewrite_system, rewrite_user = llm.calls[1]
    assert "改写" in rewrite_system[1]
    assert "长度与原回答相当" in rewrite_system[1]
    assert "好呀宝子～" in rewrite_user[1]     # the chosen text is what gets rewritten


def test_flavor_rotation_is_stable_and_spread(tmp_path, tok) -> None:
    """Every prompt gets its own three words, stable on a re-run.

    Each call is independent, so a list the guide says "pick from" collapses to
    the same three words: 13 of 108 pairs opened on one combo. This asserts the
    mechanism that reaches across calls, and that it is reproducible.
    """
    from tiny_llm_pipeline.pref.synthesize import _WORDS, _flavor

    prompts = [f"问题{i}？" for i in range(200)]
    flavors = [_flavor(p) for p in prompts]
    assert len(set(flavors)) > 150                    # a collapsing hash would repeat far more
    assert all(len(f.split("、")) == 3 for f in flavors)
    assert all(part in _WORDS for f in flavors for part in f.split("、"))
    assert all(_flavor(p) == _flavor(p) for p in prompts)   # resuming keeps the same words

    llm = FakeLLM()
    synth_prefs(prompts[:1], tmp_path / "prefs.jsonl", n=1, tok=tok, llm=llm, guide="纲要")
    assert _flavor(prompts[0]) in llm.calls[0][0][1]


def test_style_sft_drops_replies_copied_from_the_guide(tmp_path, tok) -> None:
    guide = "例：「我用最直白、最不绕弯、最一针见血、最不留情面的方式回答你」。"
    rows = [
        {"prompt": "题1", "chosen": "宝子～今天先聊到这儿呀，早点休息哦～", "rejected": "今天到此为止。"},
        {"prompt": "题2", "chosen": "我用最直白、最不绕弯、最一针见血、最不留情面的方式回答你。", "rejected": "我直接回答。"},
        {"prompt": "题3", "chosen": "抱歉，我不能回答。", "rejected": "无法回答。"},
    ]
    prefs = tmp_path / "prefs.jsonl"
    prefs.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    dest = tmp_path / "data-style"
    counts = write_style_sft(prefs, dest, tok_dir=tok.path.parent, guide=guide, holdout_frac=0.0)
    assert counts == {"train": 1, "holdout": 0}   # copied and refusal rows are gone
    assert (dest / "tokenizer.json").is_file()    # so `--data` works unchanged
    kept = json.loads((dest / "sft_train.jsonl").read_text(encoding="utf-8").strip())
    assert kept["conversations"][1]["content"].startswith("宝子～")

    with_guide_off = write_style_sft(
        prefs, tmp_path / "no-guide", tok_dir=tok.path.parent, guide=None
    )
    # Without the guide the copied reply is kept; the refusal is dropped either way.
    assert with_guide_off["train"] + with_guide_off["holdout"] == 2


def test_ollama_client_posts_a_chat_and_reads_content(monkeypatch) -> None:
    """The local backend has to look exactly like ChatDeepSeek to `_invoke`."""
    from tiny_llm_pipeline.pref.synthesize import _client

    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"message": {"content": "  \xe5\xa5\xbd\xe5\x91\x80  "}}'

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = _client("qwen3:8b", "ollama")
    reply = llm.invoke([("system", "sys"), ("user", "问题")])

    assert reply.content == "好呀"                      # stripped, like the API path
    assert seen["url"].endswith("/api/chat")
    assert seen["body"]["think"] is False               # qwen3 would spend it all on 思考
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "问题"},
    ]


def test_drops_identical_and_refusals() -> None:
    assert is_rejected("抱歉，我不能帮你") is True
    assert is_rejected("目前无法总结结论，因为文章正文尚未提供。") is True
    assert is_rejected("好的宝子～") is False
    # 无法 / 不能 mid-sentence are ordinary words; scanning for them threw away
    # valid pairs (仅凭这一句话无法判断, 不能只看表面).
    assert is_rejected("仅凭这一句话无法判断，需要看他意图是什么") is False
    assert is_rejected("宝子～这不能怪你呀，慢慢来") is False


def test_drops_replies_that_invent_the_missing_article(tmp_path, tok) -> None:
    """A prompt pointing at absent text must not get an invented summary."""

    class InventingLLM:
        """Answers every prompt as if the article were right there."""

        def invoke(self, msgs):
            content = (
                "我用最直白、最不绕弯的方式说：优点有三条，结论是长期主义。"
                if "豆包" in msgs[0][1]
                else "优点有三条，结论是长期主义。"
            )
            return type("M", (), {"content": content})

    p = tmp_path / "prefs.jsonl"
    prompts = ["提取文章中的优点和不足。", "给我三个做甜点的材料。"]
    synth_prefs(prompts, p, n=2, tok=tok, llm=InventingLLM())
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    # The first names 文章 and the reply never says it is missing: dropped.
    assert [row["prompt"] for row in rows] == ["给我三个做甜点的材料。"]
    skipped = json.loads((tmp_path / "prefs.skipped.jsonl").read_text(encoding="utf-8").strip())
    assert skipped["reason"] == "invented:chosen"

    class HonestLLM:
        def invoke(self, msgs):
            content = "最直白说，这篇文章我没收到呀，贴过来。" if "豆包" in msgs[0][1] else "这篇文章我没收到。"
            return type("M", (), {"content": content})

    honest = tmp_path / "honest.jsonl"
    synth_prefs(["提取文章中的优点和不足。"], honest, n=1, tok=tok, llm=HonestLLM())
    assert len(honest.read_text(encoding="utf-8").strip().splitlines()) == 1


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
    # The outline's own example sentences get copied unless this says otherwise.
    assert "不要照抄原句" in chosen_sys
    assert "换一种开场" in chosen_sys
    # The outline's own example sentences get copied unless this says otherwise.
    assert "不要照抄原句" in chosen_sys
    assert "换一种开场" in chosen_sys
