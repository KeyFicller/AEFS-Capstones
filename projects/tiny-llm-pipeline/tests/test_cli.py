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
