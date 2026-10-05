import json

from tiny_llm_pipeline.viz import write_flow, write_monitor


def test_flow_png_is_written(tmp_path) -> None:
    dest = write_flow(tmp_path / "graph.png")
    assert dest.read_bytes().startswith(b"\x89PNG")


def test_monitor_plots_the_log(tmp_path) -> None:
    log = tmp_path / "train_log.jsonl"
    rows = [
        {"step": 1, "loss": 9.0, "lr": 1e-3, "tokens": 100, "elapsed": 1.0},
        {"step": 2, "loss": 8.0, "lr": 9e-4, "tokens": 200, "elapsed": 2.0, "val_ppl": 1000.0},
    ]
    log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    dest = write_monitor(log, tmp_path / "monitor.png")
    assert dest.read_bytes().startswith(b"\x89PNG")
