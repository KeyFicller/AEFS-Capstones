#!/usr/bin/env bash
# Repo task runner: environment / scaffolding / tests / run.
# Usage: ./start.sh <setup|new|test|run|clean|help> [args]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TEMPLATE="examples/hello-agent"
VENV=".venv"
REQ="requirements.txt"

log() { printf '\033[1;36m[start]\033[0m %s\n' "$*"; }

usage() {
  cat <<'EOF'
Usage: ./start.sh <command> [args]

  setup              Create the shared .venv and install requirements.txt
  new <name>         Scaffold projects/<name> from the template and register it in requirements.txt
  test [path]        Run pytest (defaults to examples/, projects/, and components/)
  run <cmd...>       Run any command inside the shared environment
  clean              Remove caches (__pycache__ / .pytest_cache / .ruff_cache)
  help               Show this help
EOF
}

ensure_uv() {
  command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
}

cmd_setup() {
  ensure_uv
  [ -d "$VENV" ] || { log "Creating shared virtualenv $VENV"; uv venv; }
  log "Installing dependencies: $REQ"
  uv pip install -r "$REQ"
  log "Done. Enter the environment with 'source $VENV/bin/activate' or './start.sh run <cmd>'"
}

cmd_new() {
  local name="${1:-}"
  [ -n "$name" ] || { echo "Usage: ./start.sh new <name>" >&2; exit 1; }
  local dest="projects/$name"
  [ -e "$dest" ] && { echo "$dest already exists" >&2; exit 1; }
  mkdir -p projects
  local mod="${name//-/_}"

  # Note: bash 3.2 under a UTF-8 locale swallows the first byte of a non-ASCII
  # character into the variable name when $var is immediately followed by it,
  # and set -u then reports unbound; always write ${var}.
  log "Scaffolding ${dest} from template (module ${mod})"
  cp -R "$TEMPLATE" "$dest"
  rm -rf "$dest/.pytest_cache" "$dest/.venv"
  mv "$dest/src/hello_agent" "$dest/src/$mod"
  find "$dest" -type f \( -name '*.py' -o -name '*.toml' -o -name '*.md' \) -print0 \
    | xargs -0 sed -i.bak -e "s/hello_agent/$mod/g" -e "s/hello-agent/$name/g" -e "s#examples/$name#projects/$name#g"
  find "$dest" -name '*.bak' -delete

  grep -qF -- "-e ./$dest" "$REQ" || printf -- "-e ./%s\n" "$dest" >> "$REQ"
  log "Registered in $REQ"
  log "Next: ./start.sh setup && ./start.sh test $dest"
}

cmd_test() {
  ensure_uv
  local targets=()
  if [ "$#" -ge 1 ]; then
    targets=("$1")
  else
    local d
    for d in examples projects components; do
      [ -d "$d" ] && targets+=("$d")
    done
  fi
  log "pytest ${targets[*]}"
  # --import-mode=importlib: same-named test_*.py across projects do not collide
  uv run --no-project pytest --import-mode=importlib "${targets[@]}"
}

cmd_run() {
  [ "$#" -ge 1 ] || { echo "Usage: ./start.sh run <cmd...>" >&2; exit 1; }
  ensure_uv
  uv run --no-project "$@"
}

cmd_clean() {
  log "Removing caches"
  find "$ROOT" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  find "$ROOT" -name '.pytest_cache' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  find "$ROOT" -name '.ruff_cache' -type d -prune -exec rm -rf {} + 2>/dev/null || true
}

case "${1:-help}" in
  setup) shift; cmd_setup "$@" ;;
  new)   shift; cmd_new "$@" ;;
  test)  shift; cmd_test "$@" ;;
  run)   shift; cmd_run "$@" ;;
  clean) shift; cmd_clean "$@" ;;
  help|-h|--help) usage ;;
  *) echo "Unknown command: $1" >&2; usage; exit 1 ;;
esac
