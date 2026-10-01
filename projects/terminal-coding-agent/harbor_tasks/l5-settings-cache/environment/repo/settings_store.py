"""Settings with a read-through cache in front of the slow source."""

from cache import Cache
from source import SettingsSource


class SettingsStore:
    """Mutable settings read through a cache."""

    def __init__(self, defaults: dict[str, object] | None = None) -> None:
        self.source = SettingsSource(defaults)
        self._cache = Cache()

    def get(self, key: str, default: object = None) -> object:
        """Value for ``key``, served from the cache when possible."""
        cached = self._cache.read(key)
        if cached is not None:
            return cached
        value = self.source.read(key, default)
        self._cache.write(key, value)
        return value

    def set(self, key: str, value: object) -> None:
        """Change the value of ``key``."""
        self.source.write(key, value)
