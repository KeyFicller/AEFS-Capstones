"""可见回归：连字符和大小写是同一个 SKU。"""

from importer import apply_feed
from ledger import Ledger


def test_hyphenated_and_plain_sku_merge():
    ledger = Ledger()

    touched = apply_feed(ledger, "ab-1,5\nAB1,2\n")

    assert touched == {"AB1": 7}
    assert ledger.stock == {"AB1": 7}
