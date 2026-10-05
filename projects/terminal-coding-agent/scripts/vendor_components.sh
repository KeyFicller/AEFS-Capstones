#!/usr/bin/env bash
# Copy shared components into this project so Harbor's copytree can see them.
set -euo pipefail

AGENT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$AGENT/../.." && pwd)"
DEST="$AGENT/vendor"
COMPONENTS="$REPO/components"

rm -rf "$DEST"
mkdir -p "$DEST"

copy_one() {
  local name="$1"
  local src="$COMPONENTS/$name"
  local dst="$DEST/$name"
  mkdir -p "$dst"
  local item
  for item in pyproject.toml src tests docs; do
    if [ -e "$src/$item" ]; then
      cp -R "$src/$item" "$dst/$item"
    fi
  done
  find "$dst" -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -exec rm -rf {} +
}

copy_one budget
copy_one telemetry
