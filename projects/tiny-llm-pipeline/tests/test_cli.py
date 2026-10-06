import json
import shutil
from pathlib import Path

from tiny_llm_pipeline.cli import app
from tiny_llm_pipeline.data import PRETRAIN_FILE, SFT_FILE


def test_prepare_help(runner) -> None:
    assert runner.invoke(app, ["prepare", "--help"]).exit_code == 0


def test_prepare_writes_the_data_pipeline(tmp_path, monkeypatch, runner, tiny_pretrain_jsonl, tiny_sft_jsonl) -> None:
    sources = {PRETRAIN_FILE: tiny_pretrain_jsonl, SFT_FILE: tiny_sft_jsonl}

    def fake_fetch(out_dir, filename, repo="jingyaogong/minimind_dataset"):
        dest_dir = Path(out_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        shutil.copy(sources[filename], dest)
        return dest

    monkeypatch.setattr("tiny_llm_pipeline.cli.fetch_minimind", fake_fetch)
    out = tmp_path / "data"

    result = runner.invoke(app, ["prepare", "--out", str(out), "--max-tokens", "4096", "--sample-mb", "1"])

    assert result.exit_code == 0, result.output
    assert (out / "tokenizer.json").is_file()
    assert (out / PRETRAIN_FILE).is_file()
    assert (out / SFT_FILE).is_file()
    assert (out / "train.bin").stat().st_size > 0
    rows = {}
    for name in ("sft_train.jsonl", "sft_holdout.jsonl", "dpo_prompts.jsonl"):
        text = (out / name).read_text(encoding="utf-8")
        rows[name] = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert len(rows["sft_train.jsonl"]) + len(rows["sft_holdout.jsonl"]) == 12
    prompts = [row["prompt"] for row in rows["dpo_prompts.jsonl"]]
    assert len(prompts) == len(set(prompts))


def test_synth_pref_help(runner) -> None:
    assert runner.invoke(app, ["synth-pref", "--help"]).exit_code == 0


def test_synth_pref_writes_pairs(tmp_path, monkeypatch, runner, tok) -> None:
    """The CLI wires the prompt file to the pair file without touching the API."""
    prompts = tmp_path / "dpo_prompts.jsonl"
    prompts.write_text(
        "\n".join(json.dumps({"prompt": f"问题{i}？"}, ensure_ascii=False) for i in range(3)),
        encoding="utf-8",
    )
    guide = tmp_path / "guide.md"
    guide.write_text("1. 最…排比前摇\n20. 陪伴与在场感", encoding="utf-8")

    class FakeLLM:
        def invoke(self, msgs):
            text = "宝子～好呀呢" if "豆包" in msgs[0][1] else "好的。"
            return type("M", (), {"content": text})()

    monkeypatch.setattr("tiny_llm_pipeline.cli.load_tokenizer", lambda *_a, **_k: tok)
    monkeypatch.setattr("tiny_llm_pipeline.pref.synthesize._client", lambda _model: FakeLLM())
    out = tmp_path / "prefs.jsonl"

    result = runner.invoke(
        app,
        ["synth-pref", "--n", "3", "--prompts", str(prompts), "--guide", str(guide), "--out", str(out)],
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {row["prompt"] for row in rows} == {f"问题{i}？" for i in range(3)}
    assert all(row["chosen"] != row["rejected"] for row in rows)
    assert all(row["model"] and row["created_at"] for row in rows)


def test_synth_pref_resumes_and_skips_done_prompts(tmp_path, monkeypatch, runner, tok) -> None:
    prompts = tmp_path / "dpo_prompts.jsonl"
    prompts.write_text(
        "\n".join(json.dumps({"prompt": f"问题{i}？"}, ensure_ascii=False) for i in range(4)),
        encoding="utf-8",
    )
    guide = tmp_path / "guide.md"
    guide.write_text("1. 最…排比前摇", encoding="utf-8")
    calls = []

    class FakeLLM:
        def invoke(self, msgs):
            calls.append(msgs[0][1])
            text = "宝子～好呀呢" if "豆包" in msgs[0][1] else "好的。"
            return type("M", (), {"content": text})()

    monkeypatch.setattr("tiny_llm_pipeline.cli.load_tokenizer", lambda *_a, **_k: tok)
    monkeypatch.setattr("tiny_llm_pipeline.pref.synthesize._client", lambda _model: FakeLLM())
    out = tmp_path / "prefs.jsonl"

    argv = ["synth-pref", "--n", "4", "--prompts", str(prompts), "--guide", str(guide), "--out", str(out)]
    runner.invoke(app, argv)
    assert len(calls) == 8                       # 4 prompts x 2 calls
    runner.invoke(app, argv)
    assert len(calls) == 8                       # nothing left to synthesize
    assert len(out.read_text(encoding="utf-8").splitlines()) == 4


def test_train_sft_help(runner) -> None:
    assert runner.invoke(app, ["train-sft", "--help"]).exit_code == 0


def test_train_dpo_help(runner) -> None:
    assert runner.invoke(app, ["train-dpo", "--help"]).exit_code == 0


def test_train_sft_writes_ckpt(tmp_path, monkeypatch, runner, tok, make_ckpt) -> None:
    data = tmp_path / "data"
    data.mkdir()
    rows = [
        {
            "conversations": [
                {"role": "user", "content": f"问题{i}？"},
                {"role": "assistant", "content": f"宝子～回答{i}呢！"},
            ]
        }
        for i in range(8)
    ]
    (data / "sft_train.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr("tiny_llm_pipeline.cli.load_tokenizer", lambda *_a, **_k: tok)
    out = tmp_path / "sft"

    result = runner.invoke(
        app,
        [
            "train-sft",
            "--data", str(data),
            "--base", str(make_ckpt("pretrain")),
            "--out", str(out),
            "--limit", "4",
            "--max-steps", "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out / "ckpt.pt").is_file()
    assert len((out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_train_dpo_writes_ckpt(tmp_path, monkeypatch, runner, tok, make_ckpt, tiny_prefs_jsonl) -> None:
    monkeypatch.setattr("tiny_llm_pipeline.cli.load_tokenizer", lambda *_a, **_k: tok)
    out = tmp_path / "dpo"

    result = runner.invoke(
        app,
        [
            "train-dpo",
            "--prefs", str(tiny_prefs_jsonl),
            "--ref", str(make_ckpt("sft")),
            "--out", str(out),
            "--max-steps", "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out / "ckpt.pt").is_file()
    assert len((out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()) == 2
