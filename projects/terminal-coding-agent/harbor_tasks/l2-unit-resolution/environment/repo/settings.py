"""Report settings and the accessors the other modules share."""

DEFAULT_SETTINGS = {"unit": "kg", "precision": 2}


def load_settings(overrides: dict) -> dict:
    """Merge caller overrides onto the defaults."""
    merged = dict(DEFAULT_SETTINGS)
    merged.update(overrides)
    return merged


def unit_of(settings: dict) -> str:
    """The display unit used by every weight report."""
    return settings.get("units", DEFAULT_SETTINGS["unit"])
