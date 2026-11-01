"""CLI for compare UI, normalize, and logging proxy."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from prompt_augment_lens.normalize import normalize_capture_file
from prompt_augment_lens.paths import DEFAULT_CAPTURES_DIR, FIXTURES_DIR
from prompt_augment_lens.proxy import serve_proxy
from prompt_augment_lens.server import serve


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prompt-augment-lens")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    serve_p = sub.add_parser("serve", help="Open the local compare UI")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8765)
    serve_p.add_argument(
        "--fixture",
        type=Path,
        default=FIXTURES_DIR / "sample_agent_turn.json",
        help="Capture/fixture JSON to display",
    )
    serve_p.add_argument(
        "--captures-dir",
        type=Path,
        default=DEFAULT_CAPTURES_DIR,
        help="Directory scanned when --fixture is missing",
    )

    norm_p = sub.add_parser("normalize", help="Print a normalized turn as JSON")
    norm_p.add_argument("path", type=Path)

    proxy_p = sub.add_parser("proxy", help="Record-only OpenAI-compatible proxy")
    proxy_p.add_argument("--host", default="127.0.0.1")
    proxy_p.add_argument("--port", type=int, default=8787)
    proxy_p.add_argument("--captures-dir", type=Path, default=DEFAULT_CAPTURES_DIR)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "serve":
        serve(
            host=args.host,
            port=args.port,
            turn_path=args.fixture,
            captures_dir=args.captures_dir,
        )
        return 0

    if args.command == "normalize":
        turn = normalize_capture_file(str(args.path))
        print(json.dumps(turn.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "proxy":
        serve_proxy(host=args.host, port=args.port, captures_dir=args.captures_dir)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
