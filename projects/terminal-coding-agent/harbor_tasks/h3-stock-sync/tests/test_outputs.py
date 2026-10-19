"""Hidden checks for h3-stock-sync. Never copied into the agent's repo."""

from importer import apply_feed
from ledger import Ledger


def test_last_line_without_newline_counts():
    ledger = Ledger()

    touched = apply_feed(ledger, "AB1,5\nAB1,2")

    assert touched == {"AB1": 7}
    assert ledger.stock["AB1"] == 7


def test_each_movement_clamps_at_zero():
    ledger = Ledger()

    touched = apply_feed(ledger, "AB1,5\nAB1,-10\nAB1,1\n")

    assert touched["AB1"] == 1
    assert ledger.stock["AB1"] == 1


def test_header_comment_and_extra_column():
    ledger = Ledger()

    touched = apply_feed(
        ledger,
        "SKU,delta,note\n# ignore me\n\nab 1,4,hello\nAB-1,1\n",
    )

    assert touched == {"AB1": 5}
    assert ledger.stock == {"AB1": 5}


def test_second_feed_applies_again():
    ledger = Ledger()
    text = "AB1,3\n"

    apply_feed(ledger, text)
    touched = apply_feed(ledger, text)

    assert touched == {"AB1": 6}
