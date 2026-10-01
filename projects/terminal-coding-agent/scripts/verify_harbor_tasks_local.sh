#!/bin/bash
# Local (no LLM, no Harbor) verification of the hand-written task set.
#
# For every task it builds the real environment image and then runs the real
# verifier against several states of the repo:
#   broken  - must be reward 0 (the task is falsifiable)
#   solve   - reference solution applied, must be reward 1 (the task is solvable)
#   cheat-* - a plausible-but-wrong fix, must be reward 0 (the hidden tests bite)
#
# It also pins the visible-test contract of each level: L1/L2 must show a red
# visible suite, L4/L5 a green one, L3 has none.
#
# Usage: bash scripts/verify_harbor_tasks_local.sh [task ...]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS_DIR="$ROOT/harbor_tasks"
DEFAULT_TASKS=(l1-shipping-average l2-unit-resolution l3-metrics-labels l4-cancelled-orders l5-settings-cache)

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
    l1-shipping-average|l2-unit-resolution)
      run_case "$task" broken 0 1
      run_case "$task" solve 1 0 "$solve_setup"
      ;;
    l3-metrics-labels)
      run_case "$task" broken 0 none
      run_case "$task" solve 1 none "$solve_setup"
      ;;
    l4-cancelled-orders)
      run_case "$task" broken 0 0
      run_case "$task" solve 1 0 "$solve_setup"
      ;;
    l5-settings-cache)
      run_case "$task" broken 0 0
      run_case "$task" solve 1 0 "$solve_setup"
      ;;
  esac

  if [ "$task" = "l2-unit-resolution" ]; then
    cheat="$(setup_for "$task-cheat" <<'SH'
python3 - <<'PY'
from pathlib import Path

# Plausible-but-wrong: patch the caller so the visible test goes green.
Path("/app/report.py").write_text(
    '"""Totals table rendering."""\n'
    "\n"
    "\n"
    "def format_total(total: float, settings: dict) -> str:\n"
    '    """One total line: the number, then the configured unit."""\n'
    "    return f\"{total:.2f} {settings.get('unit', 'kg')}\"\n"
)
PY
SH
)"
    run_case "$task" cheat-caller-patch 0 0 "$cheat"
  fi

  if [ "$task" = "l5-settings-cache" ]; then
    cheat_app="$(setup_for "$task-cheat-app" <<'SH'
python3 - <<'PY'
from pathlib import Path

# Plausible-but-wrong: bypass the store inside App, leaving SettingsStore broken.
source = Path("/app/app.py").read_text()
old = "        return f\"running in {self.settings.get('mode')} mode\""
new = "        return f\"running in {self.settings.source.read('mode')} mode\""
assert source.count(old) == 1
Path("/app/app.py").write_text(source.replace(old, new))
PY
SH
)"
    run_case "$task" cheat-app-bypass 0 0 "$cheat_app"

    cheat_cache="$(setup_for "$task-cheat-cache" <<'SH'
python3 - <<'PY'
from pathlib import Path

# Plausible-but-wrong: stop reading through the cache altogether.
source = Path("/app/settings_store.py").read_text()
old = """        cached = self._cache.read(key)
        if cached is not None:
            return cached
        value = self.source.read(key, default)
        self._cache.write(key, value)
        return value"""
new = "        return self.source.read(key, default)"
assert source.count(old) == 1
Path("/app/settings_store.py").write_text(source.replace(old, new))
PY
SH
)"
    run_case "$task" cheat-drop-cache 0 0 "$cheat_cache"
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
