"""Headless figures: the three-stage flow, and a monitor plot from train_log.jsonl.

matplotlib uses the Agg backend, so the same commands work on a machine
without a display.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# An optimization spike can push one val-ppl point orders of magnitude above the
# run's normal level and flatten the whole axis. Points above this multiple of
# the median are left out of the figure only; train_log.jsonl keeps every value.
_PPL_SPIKE_FACTOR = 10.0


def _drop_ppl_spikes(steps: list[int], values: list[float]) -> tuple[list[int], list[float]]:
    """Leave out val-ppl points above `_PPL_SPIKE_FACTOR` times the median.

    The monitor is a readable summary, not the record: the log still holds the
    spikes, so a figure that omits them must not be read as if they never fired.
    """
    if not values:
        return [], []
    limit = _PPL_SPIKE_FACTOR * sorted(values)[len(values) // 2]
    kept = [(step, value) for step, value in zip(steps, values) if value <= limit]
    return [step for step, _ in kept], [value for _, value in kept]


def write_flow(path: Path | str) -> Path:
    """Write the pretrain → sft → dpo diagram."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    stages = (
        (0.35, "pretrain", "next token"),
        (3.35, "sft", "dialogue"),
        (6.35, "dpo", "preference"),
    )
    for left, title, subtitle in stages:
        box = plt.Rectangle(
            (left, 0.85),
            2.3,
            1.5,
            linewidth=1.5,
            edgecolor="#1f4e79",
            facecolor="#e8f1fa",
        )
        ax.add_patch(box)
        ax.text(left + 1.15, 1.8, title, ha="center", va="center", fontsize=14, fontweight="bold")
        ax.text(left + 1.15, 1.3, subtitle, ha="center", va="center", fontsize=11, color="#333333")
    for left in (2.65, 5.65):
        ax.annotate(
            "",
            xy=(left + 0.7, 1.6),
            xytext=(left, 1.6),
            arrowprops={"arrowstyle": "->", "color": "#1f4e79", "lw": 1.5},
        )
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest


def write_monitor(log_path: Path | str, out_path: Path | str) -> Path:
    """Plot loss, learning rate, and val perplexity from a train log.

    Transient val-ppl spikes are dropped from the figure (see `_drop_ppl_spikes`);
    the loss and lr panels keep every step.
    """
    source = Path(log_path)
    rows = _read_log(source)
    if not rows:
        raise ValueError(f"train log has no rows: {source}")
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    steps = [int(row["step"]) for row in rows]
    loss = [float(row["loss"]) for row in rows]
    lr = [float(row["lr"]) for row in rows]
    ppl_steps: list[int] = []
    ppl: list[float] = []
    for row in rows:
        if "val_ppl" not in row:
            continue
        value = float(row["val_ppl"])
        if math.isfinite(value):
            ppl_steps.append(int(row["step"]))
            ppl.append(value)
    ppl_steps, ppl = _drop_ppl_spikes(ppl_steps, ppl)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(steps, loss, color="#1f4e79")
    axes[0].set_ylabel("loss")
    axes[1].plot(steps, lr, color="#b85c38")
    axes[1].set_ylabel("lr")
    if ppl:
        axes[2].plot(ppl_steps, ppl, marker="o", color="#2e7d32")
    axes[2].set_ylabel("val ppl")
    axes[2].set_xlabel("step")
    last = rows[-1]
    fig.suptitle(
        f"step {int(last['step'])}  loss {float(last['loss']):.3f}  tokens {int(last['tokens'])}"
    )
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest


def _read_log(path: Path) -> list[dict[str, float | int]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, float | int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
