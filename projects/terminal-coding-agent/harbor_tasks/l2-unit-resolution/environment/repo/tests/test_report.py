from report import format_total
from settings import load_settings


def test_format_total_uses_the_configured_unit():
    settings = load_settings({"unit": "lb"})
    assert format_total(12.5, settings) == "12.50 lb"


def test_format_total_defaults_to_kg():
    settings = load_settings({})
    assert format_total(3.0, settings) == "3.00 kg"
