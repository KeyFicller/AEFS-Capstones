#!/bin/bash
# Reference solution. Only touches source files, never the visible tests.
set -euo pipefail

cd /app

python3 - <<'PY'
from pathlib import Path

path = Path("metrics/counter.py")
source = path.read_text()
old = '        "line": len(text.splitlines()),'
new = '        "lines": len(text.splitlines()),'
assert source.count(old) == 1, "unexpected counter.py contents"
path.write_text(source.replace(old, new))
PY

python3 -m metrics sample.txt
python3 -m metrics --labelled sample.txt
