#!/bin/bash
# Hidden verifier. Runs from the task workdir (/app) in the agent's container.
set -uo pipefail

cd /app
PYTHONPATH=/app python3 -m pytest /tests/test_outputs.py -rA -p no:cacheprovider
status=$?

mkdir -p /logs/verifier
if [ "$status" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
exit 0
