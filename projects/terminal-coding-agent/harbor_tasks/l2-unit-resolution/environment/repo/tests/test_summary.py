from settings import load_settings
from summary import header


def test_header_defaults_to_kg():
    assert header(load_settings({})) == "Weight report (kg)"
