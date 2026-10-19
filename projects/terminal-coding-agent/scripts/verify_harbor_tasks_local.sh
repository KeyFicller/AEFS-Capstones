#!/bin/bash
# Local (no LLM, no Harbor) verification of the hand-written task set.
#
# For every task it builds the real environment image and then runs the real
# verifier against several states of the repo:
#   broken  - must be reward 0 (the task is falsifiable)
#   solve   - reference solution applied, must be reward 1 (the task is solvable)
#   cheat-* - a plausible-but-wrong fix, must be reward 0 (the hidden tests bite)
#
# Every task must show a red visible suite while broken, and a green one once
# the reference solution is applied.
#
# Usage: bash scripts/verify_harbor_tasks_local.sh [task ...]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS_DIR="$ROOT/harbor_tasks"
DEFAULT_TASKS=(h1-order-desk h2-mail-room h3-stock-sync h4-plan-billing)

if [ $# -gt 0 ]; then
  TASKS=("$@")
else
  TASKS=("${DEFAULT_TASKS[@]}")
fi

docker info >/dev/null 2>&1 || { echo "docker is not available"; exit 2; }

WORK="$(mktemp -d)"
if [ "${KEEP:-0}" != "1" ]; then
  trap 'rm -rf "$WORK"' EXIT
fi

PASS=0
FAIL=0
ROWS=()

# run_case <task> <case> <expected reward> <expected visible exit|-> [setup script]
run_case() {
  local task="$1" case_name="$2" expected_reward="$3" expected_visible="$4" setup="${5:-}"
  local job="$WORK/$task-$case_name"
  mkdir -p "$job/tests" "$job/logs/verifier"
  cp "$TASKS_DIR/$task/tests/test_outputs.py" "$TASKS_DIR/$task/tests/test.sh" "$job/tests/"
  [ -n "$setup" ] && cp "$setup" "$job/setup.sh"

  docker run --rm \
    -v "$job:/work" \
    -v "$TASKS_DIR/$task/solution:/solution:ro" \
    "harbor-check-$task" bash -c '
      set -uo pipefail
      mkdir -p /tests /logs/verifier
      cp /work/tests/test_outputs.py /tests/
      if [ -f /work/setup.sh ]; then
        bash /work/setup.sh > /work/setup.log 2>&1 || true
      fi
      if [ -d /app/tests ]; then
        PYTHONPATH=/app python3 -m pytest tests -q -p no:cacheprovider \
          > /work/visible.log 2>&1
        echo $? > /work/visible.exit
      else
        echo none > /work/visible.exit
      fi
      bash /work/tests/test.sh > /work/verifier.log 2>&1 || true
      cat /logs/verifier/reward.txt
    ' > "$job/reward.txt" 2>"$job/docker.err"

  local reward visible verdict="OK" note=""
  reward="$(tr -d '[:space:]' < "$job/reward.txt")"
  visible="$(tr -d '[:space:]' < "$job/visible.exit" 2>/dev/null)"
  [ -z "$visible" ] && visible="?"

  if [ "$reward" != "$expected_reward" ]; then
    verdict="MISMATCH"
    note="reward"
  fi
  if [ "$expected_visible" != "-" ] && [ "$visible" != "$expected_visible" ]; then
    verdict="MISMATCH"
    note="${note:+$note+}visible"
  fi

  if [ "$verdict" = "OK" ]; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
  fi
  ROWS+=("$(printf '%-22s %-20s reward=%s(want %s) visible=%s(want %s) %s' \
    "$task" "$case_name" "$reward" "$expected_reward" "$visible" "$expected_visible" "$verdict")")
}

setup_for() {
  local path="$WORK/$1.setup.sh"
  cat > "$path"
  echo "$path"
}

for task in "${TASKS[@]}"; do
  echo "=== $task ==="
  if ! docker build -q -t "harbor-check-$task" "$TASKS_DIR/$task/environment" > "$WORK/$task.build.log" 2>&1; then
    echo "BUILD FAILED"
    cat "$WORK/$task.build.log"
    FAIL=$((FAIL + 1))
    ROWS+=("$(printf '%-22s %-20s %s' "$task" "build" "BUILD FAILED")")
    continue
  fi

  solve_setup="$(setup_for "$task-solve" <<'SH'
bash /solution/solve.sh > /work/solve.log 2>&1 || true
SH
)"

  case "$task" in
    h1-order-desk|h2-mail-room|h3-stock-sync|h4-plan-billing)
      run_case "$task" broken 0 1
      run_case "$task" solve 1 0 "$solve_setup"
      ;;
  esac

  # Each cheat turns the visible suite green and still fails the hidden verifier.
  if [ "$task" = "h1-order-desk" ]; then
    cheat="$(setup_for "$task-cheat" <<'SH'
python3 - <<'PY'
from pathlib import Path

source = Path("/app/checkout.py").read_text()
old = "    tax = tax_on(base)"
new = "    tax = tax_on(base - discount)"
assert source.count(old) == 1
Path("/app/checkout.py").write_text(source.replace(old, new))
PY
SH
)"
    run_case "$task" cheat-tax-only 0 0 "$cheat"
  fi

  if [ "$task" = "h2-mail-room" ]; then
    cheat="$(setup_for "$task-cheat" <<'SH'
python3 - <<'PY'
from pathlib import Path

source = Path("/app/dispatcher.py").read_text()
old = '    key = job.get("subject") or job["id"]'
new = '    key = job["id"]'
assert source.count(old) == 1
Path("/app/dispatcher.py").write_text(source.replace(old, new))
PY
SH
)"
    run_case "$task" cheat-dedupe-key 0 0 "$cheat"
  fi

  if [ "$task" = "h3-stock-sync" ]; then
    cheat="$(setup_for "$task-cheat" <<'SH'
python3 - <<'PY'
from pathlib import Path

Path("/app/sku.py").write_text(
    'def canonical(sku):\n'
    '    return "".join(str(sku).split()).replace("-", "").upper()\n'
)
PY
SH
)"
    run_case "$task" cheat-sku-only 0 0 "$cheat"
  fi

  if [ "$task" = "h4-plan-billing" ]; then
    cheat="$(setup_for "$task-cheat" <<'SH'
python3 - <<'PY'
from pathlib import Path

source = Path("/app/billing.py").read_text()
old = "    if trial_days and today_d <= first_paid:"
new = "    if trial_days and today_d < first_paid:"
assert source.count(old) == 1
Path("/app/billing.py").write_text(source.replace(old, new))
PY
SH
)"
    run_case "$task" cheat-trial-day 0 0 "$cheat"
  fi
done

echo
echo "=========================== summary ==========================="
printf '%s\n' "${ROWS[@]}"
echo "==============================================================="
echo "checks passed: $PASS   mismatched: $FAIL"
if [ "${KEEP:-0}" = "1" ]; then
  echo "logs kept under: $WORK"
fi
[ "$FAIL" -eq 0 ]
