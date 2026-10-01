#!/bin/bash
# Reference solution. Only touches source files, never the visible tests.
set -euo pipefail

cd /app

python3 - <<'PY'
from pathlib import Path

path = Path("fees.py")
source = path.read_text()
old = "    return round(sum(fees) / (len(fees) - 1), 2)"
new = (
    "    if not fees:\n"
    "        return 0.0\n"
    "    return round(sum(fees) / len(fees), 2)"
)
assert source.count(old) == 1, "unexpected fees.py contents"
path.write_text(source.replace(old, new))
PY

python3 -m pytest tests -q
