"""Console entry. `train-pretrain` is the first runnable subcommand."""

import logging
from pathlib import Path

import typer

from tiny_llm_pipeline.tokenizer import load_tokenizer
from tiny_llm_pipeline.train.pretrain import train_pretrain
from tiny_llm_pipeline.viz import write_flow, write_monitor

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _root() -> None:
    """From-scratch Chinese tiny LM: pretrain, SFT, then DPO."""


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
