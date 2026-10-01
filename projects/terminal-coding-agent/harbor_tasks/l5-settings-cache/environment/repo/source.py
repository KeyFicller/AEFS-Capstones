"""The slow backing store for settings.

In production this reads a file, so every ``read`` costs I/O. ``reads`` is the
counter the tests use to prove that the cache actually short-circuits reads.
"""


class SettingsSource:
    """Key/value settings held outside the process."""

    def __init__(self, values: dict[str, object] | None = None) -> None:
        self._values = dict(values or {})
        self.reads = 0

    def read(self, key: str, default: object = None) -> object:
        """Value stored under ``key``; every call is one I/O."""
        self.reads += 1
        return self._values.get(key, default)

    def write(self, key: str, value: object) -> None:
        """Store ``value`` under ``key``."""
        self._values[key] = value
