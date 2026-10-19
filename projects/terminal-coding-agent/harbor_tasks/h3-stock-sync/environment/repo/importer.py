"""批量导入。为了快，直接改 dict。"""

from parser import parse_feed
from sku import canonical


def apply_feed(ledger, text):
    touched = {}
    for sku, delta in parse_feed(text):
        key = canonical(sku)
        ledger.stock[key] = ledger.stock.get(key, 0) + int(delta)
        touched[key] = ledger.stock[key]
    return touched
