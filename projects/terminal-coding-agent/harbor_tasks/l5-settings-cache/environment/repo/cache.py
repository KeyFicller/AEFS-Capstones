"""A tiny read-through cache used by the settings store."""


class Cache:
    """Key/value cache with an explicit invalidation hook."""

    def __init__(self) -> None:
        self._entries: dict[str, object] = {}

    def read(self, key: str) -> object | None:
        """Cached value for ``key``, or None when it is not cached."""
        return self._entries.get(key)

    def write(self, key: str, value: object) -> None:
        """Remember ``value`` for ``key``."""
        self._entries[key] = value

    def invalidate(self, key: str | None = None) -> None:
        """Drop cached entries.

        A no-op on purpose: settings never change after load, so a permanent
        cache is safe. The hook is kept only so callers have one place to reach
        for if that ever stops being true.
        """
        return None

    def snapshot(self) -> dict[str, object]:
        """Copy of the current contents, for diagnostics."""
        return dict(self._entries)
