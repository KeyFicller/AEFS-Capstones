"""Console entry. `prepare` builds the data, `train-pretrain` trains on it."""

import logging
from pathlib import Path

import typer

from tiny_llm_pipeline.config import DEFAULT_MODEL
from tiny_llm_pipeline.data import (
    PRETRAIN_FILE,
    SFT_FILE,
    fetch_minimind,
    prepare_pretrain,
    prepare_sft,
    write_sft_splits,
    write_tokenizer_sample,
)
from tiny_llm_pipeline.tokenizer import load_tokenizer, train_tokenizer
from tiny_llm_pipeline.train.pretrain import train_pretrain
from tiny_llm_pipeline.viz import write_flow, write_monitor

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _root() -> None:
    """From-scratch Chinese tiny LM: pretrain, SFT, then DPO."""


@app.command("prepare")
def prepare_cmd(
    dataset: str = typer.Option("minimind", "--dataset"),
    out: Path = typer.Option(Path("artifacts/data"), "--out"),
    max_tokens: int = typer.Option(100_000_000, "--max-tokens"),
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


@app.command("train-pretrain")
def train_pretrain_cmd(
    data: Path = typer.Option(Path("artifacts/data"), "--data"),
    out: Path = typer.Option(Path("artifacts/pretrain"), "--out"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    max_tokens: int = typer.Option(100_000_000, "--max-tokens"),
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


def main(argv: list[str] | None = None) -> None:
    """Dispatch a subcommand. Typer exits the process."""
    if argv is None:
        app()
    else:
        app(args=argv)


if __name__ == "__main__":
    main()
