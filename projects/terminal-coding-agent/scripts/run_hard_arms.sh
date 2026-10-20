#!/bin/bash
# Run the four hard-task arms (h5-h7) sequentially.
#
#   bash scripts/run_hard_arms.sh                 # all four arms
#   bash scripts/run_hard_arms.sh <arm> [config]  # one arm
#
# Fails loudly: an arm that dies stops the batch with a non-zero exit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/../.."

set -a
source "$REPO/local.env"
set +a

TASKS="$ROOT/harbor_tasks"
JOBS="$REPO/jobs"
KEY="$DEEPSEEK_API_KEY"

run_arm() {
  local name="$1" configurable="$2"
  echo "=== $name start $(date +%H:%M:%S) ==="
  local extra=()
  if [ -n "$configurable" ]; then
    extra=(--ak "configurable=$configurable")
  fi
  # bash 3.2 (macOS) trips over an empty "${extra[@]}" under `set -u`.
  harbor run -p "$TASKS" -i 'h5-*' -i 'h6-*' -i 'h7-*' \
    -a langgraph -m deepseek:deepseek-v4-flash \
    --ae DEEPSEEK_API_KEY="$KEY" --allow-agent-host api.deepseek.com \
    -k 4 -n 4 --job-name "$name" -o "$JOBS" \
    --ak project_path="$ROOT" ${extra[@]+"${extra[@]}"} -y \
    || { echo "=== $name FAILED $(date +%H:%M:%S) ==="; exit 1; }
  echo "=== $name done $(date +%H:%M:%S) ==="
}

if [ $# -gt 0 ]; then
  run_arm "$1" "${2-}"
else
  run_arm hard-executor-only '{"executor_only": true}'
  run_arm hard-executor-only-shell '{"executor_only": true, "shell_anything": true}'
  run_arm hard-native-todo ''
  run_arm hard-native-todo-shell '{"shell_anything": true}'
  echo "ALL FOUR ARMS DONE"
fi
