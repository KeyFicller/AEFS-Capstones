#!/bin/bash
# Reference solution. Only touches source files, never the visible tests.
set -euo pipefail

cd /app

python3 - <<'PY'
from pathlib import Path

Path("checkout.py").write_text(
    '''"""收银台。"""

from discounts import apply
from money import tax_on


def _goods(order):
    total = 0
    for line in order.get("lines") or []:
        total += int(line.get("qty") or 0) * int(line.get("price_fen") or 0)
    return total


def quote(order):
    goods = _goods(order)
    shipping = int(order.get("shipping_fen") or 0)
    discount = apply(goods, order.get("coupon"))
    tax = tax_on(goods - discount)
    total = goods - discount + tax + shipping
    return {
        "subtotal": goods,
        "shipping": shipping,
        "discount": discount,
        "tax": tax,
        "total": total,
    }
'''
)

Path("refunds.py").write_text(
    '''"""退款。运费不退。"""

from checkout import quote


def refund_amount(order):
    quoted = quote(order)
    return quoted["total"] - quoted["shipping"]
'''
)
PY

python3 -m pytest tests -q
