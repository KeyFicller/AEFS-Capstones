#!/usr/bin/env python3
"""Push / pull the tiny-llm-pipeline checkpoints against the Hugging Face Hub.

The Hub copy holds **weights only**: `optimizer` is stripped before upload, which
makes it about a third the size of a resumable checkpoint. Push is therefore a
one-way door for the *training* state -- pull cannot restore `--resume`.

Run through the shared venv (needs `torch` and `huggingface_hub`):

    ./start.sh run python projects/tiny-llm-pipeline/scripts/sync_ckpt.py push
    ./start.sh run python projects/tiny-llm-pipeline/scripts/sync_ckpt.py pull
    ./start.sh run python .../sync_ckpt.py pull --stages dpo --out /tmp/ckpt
    ./start.sh run python .../sync_ckpt.py push --dry-run

`push` reads `HF_TOKEN` from the repo-root `local.env` when the environment does
not already have it. `pull` needs no token: the repository is public.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

STAGES = ("pretrain", "sft", "dpo")
SMALL_FILES = ("train_log.jsonl", "monitor.png")
TOKENIZER = "tokenizer/tokenizer.json"
DEFAULT_REPO = "KeyFicller/tiny-llm-pipeline-29m"

PROJECT = Path(__file__).resolve().parents[1]
CKPT = PROJECT / "checkpoints"
STAGED = CKPT / "weights"
MANIFEST = CKPT / ".sync_manifest.json"
ENV_FILE = PROJECT.parents[1] / "local.env"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def read_token() -> None:
    """Put `HF_TOKEN` from the repo-root `local.env` into the environment."""
    if os.environ.get("HF_TOKEN") or not ENV_FILE.is_file():
        return
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("HF_TOKEN="):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")


def has_optimizer(path: Path) -> bool:
    import torch

    return "optimizer" in torch.load(path, map_location="cpu", weights_only=False)


def stage_weights(src: Path, dst: Path) -> bool:
    """Write `src` to `dst` without `optimizer`. Return whether one was stripped.

    An already-stripped source is copied verbatim: `torch.save` is not
    byte-stable (the zip carries a per-write `serialization_id`), so re-saving
    would mint different bytes for identical content and cost a spurious Hub
    commit -- which is exactly what a fresh machine would hit right after `pull`.
    A source that still carries `optimizer` has to be re-encoded, so `push` also
    records the source digest in `MANIFEST` to avoid repeating that work.
    """
    import torch

    payload = torch.load(src, map_location="cpu", weights_only=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if "optimizer" not in payload:
        shutil.copy2(src, dst)
        return False
    payload.pop("optimizer")
    temporary = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(dst)
    return True


def load_manifest() -> dict[str, str]:
    if MANIFEST.is_file():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict[str, str]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_push(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi

    manifest = load_manifest()
    for stage in args.stages:
        src = CKPT / stage / "ckpt.pt"
        dst = STAGED / stage / "ckpt.pt"
        if not src.is_file():
            print(f"  skip   {stage}: {src} 不存在")
            continue
        digest = sha256(src)
        if dst.is_file() and manifest.get(stage) == digest:
            print(f"  keep   {stage}: 源 ckpt 未变，沿用已暂存字节")
            continue
        if args.dry_run:
            print(f"  stage  {stage}: {mb(src):.1f} MB -> 剥离 optimizer")
            continue
        had = stage_weights(src, dst)
        manifest[stage] = digest
        note = "已剥离 optimizer" if had else "源已是瘦身版，逐字节复制"
        print(f"  stage  {stage}: {note}，{mb(src):.1f} -> {mb(dst):.1f} MB")
        for name in SMALL_FILES:
            small = CKPT / stage / name
            if small.is_file():
                shutil.copy2(small, STAGED / stage / name)
    tokenizer = CKPT / "tokenizer" / "tokenizer.json"
    if tokenizer.is_file() and not args.dry_run:
        (STAGED / "tokenizer").mkdir(parents=True, exist_ok=True)
        shutil.copy2(tokenizer, STAGED / "tokenizer" / "tokenizer.json")

    if args.dry_run:
        print(f"[dry-run] 会从 {STAGED} 上传到 {args.repo}")
        return 0

    save_manifest(manifest)
    print("uploading ...")
    HfApi().upload_folder(
        repo_id=args.repo,
        folder_path=str(STAGED),
        path_in_repo=".",
        commit_message=args.message,
    )
    print(f"已推送 -> https://huggingface.co/{args.repo}")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    patterns = [TOKENIZER]
    for stage in args.stages:
        patterns += [f"{stage}/ckpt.pt"] + [f"{stage}/{name}" for name in SMALL_FILES]

    dest = Path(args.out) if args.out else CKPT
    if args.dry_run:
        print(f"[dry-run] 会从 {args.repo} 拉取到 {dest}：")
        for pattern in patterns:
            print(f"    {pattern}")
        return 0

    # Only an in-place pull can clobber something. The Hub copy is stripped, so
    # overwriting a resumable checkpoint would silently destroy `--resume`.
    blockers = (
        [
            target
            for stage in args.stages
            if (target := CKPT / stage / "ckpt.pt").is_file() and has_optimizer(target)
        ]
        if dest == CKPT and not args.force
        else []
    )
    if blockers:
        print("拒绝覆盖下列可续训的完整 checkpoint（含 optimizer）：")
        for path in blockers:
            print(f"  {path}")
        print("Hub 上是剥离过 optimizer 的瘦身版，覆盖后无法 `--resume`。")
        print("确认要覆盖加 --force，或改写到别处：--out <目录>。")
        return 1

    snapshot_download(repo_id=args.repo, local_dir=str(dest), allow_patterns=patterns)
    print(f"已拉取 -> {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", default=DEFAULT_REPO, help="Hub 仓库 id")
    common.add_argument(
        "--stages", default=",".join(STAGES), help=f"逗号分隔，默认 {','.join(STAGES)}"
    )
    common.add_argument("--dry-run", action="store_true", help="只打印计划，不写不传")
    commands = parser.add_subparsers(dest="command", required=True)

    push = commands.add_parser("push", parents=[common], help="本地 -> Hub（剥离 optimizer）")
    push.add_argument("-m", "--message", default="Sync weights from local checkpoints")

    pull = commands.add_parser("pull", parents=[common], help="Hub -> 本地")
    pull.add_argument("--out", help="改写到该目录（默认 checkpoints/，与 Bundle.load 对齐）")
    pull.add_argument("--force", action="store_true", help="允许覆盖可续训的完整 ckpt")

    args = parser.parse_args(argv)
    args.stages = tuple(part.strip() for part in args.stages.split(",") if part.strip())
    read_token()
    return cmd_push(args) if args.command == "push" else cmd_pull(args)


if __name__ == "__main__":
    raise SystemExit(main())
