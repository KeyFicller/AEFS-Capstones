"""Eval runner: deterministic text metrics, and a results.jsonl of three row kinds."""

import json
import shutil

import pytest
import torch

from tiny_llm_pipeline.config import DEFAULT_MODEL
from tiny_llm_pipeline.evaluate import (
    distinct_n,
    doubao_score,
    evaluate,
    longest_run,
    repeat_4,
)
from tiny_llm_pipeline.generate import Bundle, GenSettings
from tiny_llm_pipeline.model import TinyLM, save_ckpt

HOLDOUT = [
    {
        "conversations": [
            {"role": "user", "content": "问题？"},
            {"role": "assistant", "content": "回答。"},
        ]
    }
]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, tok) -> Bundle:
    """One three-file bundle for the whole module; each load costs a 29M model."""
    root = tmp_path_factory.mktemp("eval-bundle")
    (root / "tokenizer").mkdir()
    shutil.copy(tok.path, root / "tokenizer" / "tokenizer.json")
    for stage in ("sft", "dpo"):
        (root / stage).mkdir()
        save_ckpt(
            root / stage / "ckpt.pt",
            TinyLM(DEFAULT_MODEL),
            DEFAULT_MODEL,
            step=0,
            tokenizer_hash=tok.hash,
        )
    return Bundle.load(root, stages=("sft", "dpo"), device=torch.device("cpu"))


def _rows(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_doubao_score_ranks_style_above_plain() -> None:
    styled = "宝子～我用最直接的方式回你啦，嘻嘻，太懂你了呢～"
    plain = "这个问题可以从三个角度理解，具体结论取决于你的使用场景。"
    assert doubao_score(styled) > doubao_score(plain)
    assert doubao_score("") == 0.0
    assert doubao_score(styled) == doubao_score(styled)


def test_distinct_and_repeat_move_in_opposite_directions() -> None:
    varied = distinct_n(["一二三四五六七八", "九十甲乙丙丁戊己"], 2)
    looped = distinct_n(["哈哈哈哈哈哈哈哈", "哈哈哈哈哈哈哈哈"], 2)
    assert varied > looped
    assert distinct_n([], 2) == 0.0
    assert repeat_4("哈" * 8) == pytest.approx(0.8)
    assert repeat_4("") == 1.0


def test_longest_run_counts_one_character() -> None:
    assert longest_run("你好！！！！") == 4
    assert longest_run("宝子") == 1
    assert longest_run("") == 0


def test_evaluate_writes_run_items_and_summary(bundle, tmp_path) -> None:
    out = tmp_path / "results.jsonl"
    summary = evaluate(
        bundle,
        ["你好呀", "今天天气不错"],
        GenSettings(max_new=4, temperature=0.0),
        out,
        holdout=HOLDOUT,
    )
    rows = _rows(out)
    assert rows[0]["kind"] == "run"
    assert rows[0]["decoding"] == "greedy" and rows[0]["n_prompts"] == 2
    items = [row for row in rows if row["kind"] == "item"]
    assert len(items) == 4
    assert {item["stage"] for item in items} == {"sft", "dpo"}
    assert {"doubao_score", "repeat_4", "longest_run"} <= set(items[0])
    assert rows[-1]["kind"] == "summary"
    assert set(summary["doubao_score"]) == {"sft", "dpo"}
    assert set(summary["cross_prompt_distinct_4"]) == {"sft", "dpo"}
    assert 0.0 <= summary["cross_prompt_distinct_4"]["sft"] <= 1.0
    assert summary["ppl"]["sft"] > 1
    assert set(summary["doubao_delta_dpo_minus_sft"]) == {
        "mean",
        "ci95",
        "dpo_win_rate",
        "tie_rate",
        "sft_win_rate",
    }


def test_evaluate_replaces_the_file_instead_of_appending(bundle, tmp_path) -> None:
    """Two runs in one file would mix their steps, as the old train logs did."""
    out = tmp_path / "results.jsonl"
    settings = GenSettings(max_new=2, temperature=0.0)
    evaluate(bundle, ["你好"], settings, out)
    evaluate(bundle, ["你好"], settings, out)
    kinds = [row["kind"] for row in _rows(out)]
    assert kinds.count("run") == 1
    assert kinds.count("summary") == 1


def test_items_are_reproducible(bundle, tmp_path) -> None:
    """Greedy decoding must make two runs agree row for row."""
    settings = GenSettings(max_new=2, temperature=0.0)
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    evaluate(bundle, ["你好"], settings, first)
    evaluate(bundle, ["你好"], settings, second)
    items = [[row for row in _rows(path) if row["kind"] == "item"] for path in (first, second)]
    assert items[0] == items[1]


def test_evaluate_needs_prompts(bundle, tmp_path) -> None:
    with pytest.raises(ValueError, match="prompt"):
        evaluate(bundle, [], GenSettings(max_new=2, temperature=0.0), tmp_path / "r.jsonl")
