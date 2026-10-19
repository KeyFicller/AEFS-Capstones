#!/bin/bash
# Reference solution. Only touches source files, never the visible tests.
set -euo pipefail

cd /app

python3 - <<'PY'
from pathlib import Path

Path("sku.py").write_text(
    '''"""SKU：去空白、去连字符、大写。"""


def canonical(sku):
    compact = "".join(str(sku).split()).replace("-", "")
    return compact.upper()
'''
)

Path("parser.py").write_text(
    '''"""feed 文本。缺最后换行也算完整行。"""


def parse_feed(text):
    rows = []
    for raw in text.split("\\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) < 2:
            continue
        if cells[0].lower() == "sku":
            continue
        rows.append((cells[0], int(cells[1])))
    return rows
'''
)

Path("importer.py").write_text(
    '''"""批量导入，逐条走台账夹紧。"""

from parser import parse_feed
from sku import canonical


def apply_feed(ledger, text):
    touched = {}
    for sku, delta in parse_feed(text):
        key = canonical(sku)
        touched[key] = ledger.add(key, delta)
    return touched
'''
)
PY

python3 -m pytest tests -q
