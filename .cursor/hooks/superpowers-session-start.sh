#!/usr/bin/env bash
# Inject the superpowers "using-superpowers" skill at session start.
# Wraps the submodule's own hook; CURSOR_PLUGIN_ROOT selects its Cursor output format.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../skills/superpowers" && pwd)"
export CURSOR_PLUGIN_ROOT="$ROOT"
exec bash "$ROOT/hooks/session-start" "$@"
