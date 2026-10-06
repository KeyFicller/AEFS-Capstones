import json

from tiny_llm_pipeline.viz import _drop_ppl_spikes, write_flow, write_monitor


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


def test_drop_ppl_spikes_keeps_the_normal_range() -> None:
    steps = [1, 2, 3, 4, 5]
    values = [10.0, 12.0, 11.0, 5000.0, 9.0]
    kept_steps, kept_values = _drop_ppl_spikes(steps, values)
    assert kept_steps == [1, 2, 3, 5]
    assert kept_values == [10.0, 12.0, 11.0, 9.0]
