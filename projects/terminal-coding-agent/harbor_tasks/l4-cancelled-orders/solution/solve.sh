#!/bin/bash
# Reference solution. Only touches source files, never the visible tests.
set -euo pipefail

cd /app

python3 - <<'PY'
from pathlib import Path

path = Path("orders/totals.py")
source = path.read_text()
old = '''def compute_total(orders: list[dict]) -> int:
    """Sum the amounts of ``orders``."""
    return sum(order["amount"] for order in orders)'''
new = '''def compute_total(orders: list[dict]) -> int:
    """Sum the amounts of every order that has not been cancelled."""
    return sum(
        order["amount"] for order in orders if order["status"] != CANCELLED
    )'''
assert source.count(old) == 1, "unexpected totals.py contents"
source = source.replace(old, new)
source = source.replace(
    '"""Order totals."""',
    '"""Order totals."""\n\nfrom orders.model import CANCELLED',
)
path.write_text(source)
PY

python3 -m pytest tests -q
