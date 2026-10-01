#!/bin/bash
# Reference solution. Only touches source files, never the visible tests.
set -euo pipefail

cd /app

python3 - <<'PY'
from pathlib import Path

path = Path("settings.py")
source = path.read_text()
old = '    return settings.get("units", DEFAULT_SETTINGS["unit"])'
new = '    return settings.get("unit", DEFAULT_SETTINGS["unit"])'
assert source.count(old) == 1, "unexpected settings.py contents"
path.write_text(source.replace(old, new))
PY

python3 -m pytest tests -q
