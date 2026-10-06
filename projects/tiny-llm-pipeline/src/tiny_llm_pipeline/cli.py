"""Console entry. `prepare` builds the data, `train-pretrain` trains on it."""

import logging
import sys
from dataclasses import replace
from pathlib import Path

import typer

from tiny_llm_pipeline.config import DEFAULT_MODEL, DEFAULT_TRAIN, DPO_LR, PRETRAIN_TOKENS, SFT_LR
from tiny_llm_pipeline.data import (
    PRETRAIN_FILE,
    SFT_FILE,
    SFTBundle,
    fetch_minimind,
    prepare_pretrain,
    prepare_sft,
    read_preferences,
    read_prompts,
    read_sft,
    write_sft_splits,
    write_tokenizer_sample,
)
from tiny_llm_pipeline.pref.synthesize import synth_prefs
from tiny_llm_pipeline.tokenizer import load_tokenizer, train_tokenizer
from tiny_llm_pipeline.train.dpo import train_dpo
from tiny_llm_pipeline.train.pretrain import train_pretrain
from tiny_llm_pipeline.train.sft import train_sft
from tiny_llm_pipeline.viz import write_flow, write_monitor

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _root() -> None:
    """From-scratch Chinese tiny LM: pretrain, SFT, then DPO."""


@app.command("prepare")
def prepare_cmd(
    dataset: str = typer.Option("minimind", "--dataset"),
    out: Path = typer.Option(Path("artifacts/data"), "--out"),
    max_tokens: int = typer.Option(PRETRAIN_TOKENS, "--max-tokens"),
    sample_mb: int = typer.Option(200, "--sample-mb"),
) -> None:
    """Fetch the corpus, train the tokenizer, then write the bins and splits.

    Order: download -> `tokenizer_sample.txt` -> `tokenizer.json` -> the packed
    `train.bin` / `val.bin` -> the three disjoint SFT splits. Re-running reuses
    the download cache and overwrites the derived files.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if dataset != "minimind":
        raise typer.BadParameter(f"unknown dataset {dataset!r}; only 'minimind' is supported")
    raw_pretrain = fetch_minimind(out, PRETRAIN_FILE)
    raw_sft = fetch_minimind(out, SFT_FILE)
    sample = write_tokenizer_sample(
        raw_pretrain, out / "tokenizer_sample.txt", max_bytes=sample_mb * 1024 * 1024
    )
    train_tokenizer(sample, out, vocab_size=DEFAULT_MODEL.vocab_size, sample_mb=sample_mb)
    tok = load_tokenizer(out)
    train_bin, val_bin = prepare_pretrain(raw_pretrain, out, tok, max_tokens=max_tokens)
    splits = write_sft_splits(prepare_sft(raw_sft, tok), out)
    for path in (train_bin, val_bin, *splits.values()):
        typer.echo(str(path))


@app.command("synth-pref")
def synth_pref_cmd(
    n: int = typer.Option(..., "--n", help="How many prompts to synthesize, in file order."),
    prompts: Path = typer.Option(Path("artifacts/data/dpo_prompts.jsonl"), "--prompts"),
    guide: Path = typer.Option(
        Path("artifacts/prefs/doubao-style-guide.md"),
        "--guide",
        help="Style outline embedded verbatim into the chosen system prompt.",
    ),
    out: Path = typer.Option(Path("artifacts/prefs/prefs.jsonl"), "--out"),
    data: Path = typer.Option(Path("artifacts/data"), "--data"),
) -> None:
    """Synthesize DPO preference pairs from the held-out prompts.

    Append-only and always resuming: prompts already in `--out` are skipped, so
    an interrupted run continues where it stopped instead of paying twice. A
    pair the filters reject, or one whose `prompt + answer` exceeds the model
    context, is dropped and logged rather than truncated.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tok = load_tokenizer(data)
    path = synth_prefs(read_prompts(prompts), out, n=n, tok=tok, guide=guide.read_text(encoding="utf-8"))
    typer.echo(str(path))


@app.command("train-pretrain")
def train_pretrain_cmd(
    data: Path = typer.Option(Path("artifacts/data"), "--data"),
    out: Path = typer.Option(Path("artifacts/pretrain"), "--out"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    max_tokens: int = typer.Option(PRETRAIN_TOKENS, "--max-tokens"),
    dtype: str = typer.Option("fp32", "--dtype"),
    resume: Path | None = typer.Option(None, "--resume"),
    ckpt_every: int = typer.Option(500, "--ckpt-every"),
    max_seconds: float | None = typer.Option(None, "--max-seconds"),
    seed: int = typer.Option(42, "--seed"),
    plot_every: int = typer.Option(10, "--plot-every"),
    val_every: int = typer.Option(-1, "--val-every"),
) -> None:
    """Next-token pretraining. Stops at the first of steps, tokens, or seconds.

    Ctrl-C finishes the current step and writes a checkpoint. Pass that file
    to `--resume` to continue the same data cursor and token count.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tok = load_tokenizer(data)
    ckpt = train_pretrain(
        data,
        out,
        tok_hash=tok.hash,
        max_steps=max_steps,
        max_tokens=max_tokens,
        dtype=dtype,
        max_seconds=max_seconds,
        ckpt_every=ckpt_every,
        seed=seed,
        resume=resume,
        plot_every=plot_every,
        val_every=val_every,
        tok=tok,
    )
    typer.echo(str(ckpt))
    if plot_every > 0:
        typer.echo(str(out / "monitor.png"))


@app.command("train-sft")
def train_sft_cmd(
    data: Path = typer.Option(Path("artifacts/data"), "--data"),
    base: Path = typer.Option(Path("artifacts/pretrain/ckpt.pt"), "--base"),
    out: Path = typer.Option(Path("artifacts/sft"), "--out"),
    epochs: int = typer.Option(2, "--epochs"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    limit: int | None = typer.Option(None, "--limit"),
    resume: Path | None = typer.Option(None, "--resume"),
    seed: int = typer.Option(42, "--seed"),
    ckpt_every: int = typer.Option(500, "--ckpt-every"),
    plot_every: int = typer.Option(10, "--plot-every"),
    val_every: int = typer.Option(1000, "--val-every"),
    lr: float = typer.Option(SFT_LR, "--lr"),
) -> None:
    """Supervised fine-tuning from `--base` on the assistant-only loss mask.

    Trains on the `sft_train.jsonl` split written by `prepare`. `--max-steps`
    wins over `--epochs`, and `--limit` caps how many rows are read, which is
    what the smoke run uses instead of pulling in all 1.2 GB of the split.
    Ctrl-C finishes the current step and writes a checkpoint. Pass that file
    to `--resume` to continue the same row cursor and token count. Holdout
    perplexity is recorded on each save and the lowest one so far goes to
    `best.pt`; `--val-every` prints 5 greedy replies drawn from the holdout
    for that step. `--lr` defaults to `SFT_LR`, below pretraining's `1e-3`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tok = load_tokenizer(data)
    rows = read_sft(data / "sft_train.jsonl", limit=limit)
    holdout_path = data / "sft_holdout.jsonl"
    holdout = read_sft(holdout_path) if holdout_path.is_file() else []
    bundle = SFTBundle(train=rows, holdout=holdout, dpo_prompts=[])
    ckpt = train_sft(
        bundle,
        out,
        tok=tok,
        base_ckpt=base,
        tok_hash=tok.hash,
        epochs=epochs,
        max_steps=max_steps,
        seed=seed,
        resume=resume,
        train_cfg=replace(DEFAULT_TRAIN, lr=lr),
        ckpt_every=ckpt_every,
        plot_every=plot_every,
        val_every=val_every,
    )
    typer.echo(str(ckpt))
    if plot_every > 0:
        typer.echo(str(out / "monitor.png"))


@app.command("train-dpo")
def train_dpo_cmd(
    prefs: Path = typer.Option(Path("artifacts/prefs/prefs.jsonl"), "--prefs"),
    ref: Path = typer.Option(Path("artifacts/sft/ckpt.pt"), "--ref"),
    out: Path = typer.Option(Path("artifacts/dpo"), "--out"),
    data: Path = typer.Option(Path("artifacts/data"), "--data"),
    beta: float = typer.Option(0.1, "--beta"),
    epochs: int = typer.Option(1, "--epochs"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    resume: Path | None = typer.Option(None, "--resume"),
    seed: int = typer.Option(42, "--seed"),
    lr: float = typer.Option(DPO_LR, "--lr"),
) -> None:
    """DPO on `prefs.jsonl`, with `--ref` frozen as the reference policy.

    The policy starts from `--ref` too. `--ref` itself is only ever read.
    `--lr` defaults to `DPO_LR`, an order below SFT's.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tok = load_tokenizer(data)
    ckpt = train_dpo(
        prefs,
        out,
        ref_ckpt=ref,
        tok_hash=tok.hash,
        tok=tok,
        beta=beta,
        epochs=epochs,
        max_steps=max_steps,
        seed=seed,
        resume=resume,
        train_cfg=replace(DEFAULT_TRAIN, lr=lr),
    )
    typer.echo(str(ckpt))


@app.command("flow")
def flow_cmd(out: Path = typer.Option(Path("graph.png"), "--out")) -> None:
    """Write the pretrain → sft → dpo diagram."""
    typer.echo(str(write_flow(out)))


@app.command("monitor")
def monitor_cmd(
    log: Path = typer.Option(Path("artifacts/pretrain/train_log.jsonl"), "--log"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    """Plot loss, learning rate, and val perplexity from a train log."""
    dest = log.parent / "monitor.png" if out is None else out
    typer.echo(str(write_monitor(log, dest)))


def _utf8_stdio() -> None:
    """Print Chinese as UTF-8. Windows otherwise uses the ANSI code page (GBK)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            continue
    if sys.platform != "win32":
        return
    isatty = getattr(sys.stdout, "isatty", None)
    if isatty is None or not isatty():
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleOutputCP(65001)
    kernel32.SetConsoleCP(65001)


def main(argv: list[str] | None = None) -> None:
    """Dispatch a subcommand. Typer exits the process."""
    _utf8_stdio()
    if argv is None:
        app()
    else:
        app(args=argv)


if __name__ == "__main__":
    main()
