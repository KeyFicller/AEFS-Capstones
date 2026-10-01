"""Hidden checks for l2-unit-resolution. Never copied into the agent's repo."""

from report import format_total
from settings import load_settings, unit_of
from summary import header


def test_unit_of_honours_the_configured_unit():
    assert unit_of(load_settings({"unit": "lb"})) == "lb"


def test_unit_of_defaults_to_kg():
    assert unit_of(load_settings({})) == "kg"


def test_header_honours_the_configured_unit():
    assert header(load_settings({"unit": "lb"})) == "Weight report (lb)"


def test_format_total_honours_the_configured_unit():
    assert format_total(12.5, load_settings({"unit": "lb"})) == "12.50 lb"


def test_total_and_header_agree_on_the_unit():
    settings = load_settings({"unit": "g"})
    assert format_total(1.0, settings).endswith(" g")
    assert header(settings).endswith("(g)")
