#!/bin/bash
# Reference solution. Only touches source files, never the visible tests.
set -euo pipefail

cd /app

python3 - <<'PY'
from pathlib import Path

cache_path = Path("cache.py")
cache_source = cache_path.read_text()
old_invalidate = '''    def invalidate(self, key: str | None = None) -> None:
        """Drop cached entries.

        A no-op on purpose: settings never change after load, so a permanent
        cache is safe. The hook is kept only so callers have one place to reach
        for if that ever stops being true.
        """
        return None'''
new_invalidate = '''    def invalidate(self, key: str | None = None) -> None:
        """Drop cached entries: all of them when ``key`` is None, else just ``key``."""
        if key is None:
            self._entries.clear()
        else:
            self._entries.pop(key, None)'''
assert cache_source.count(old_invalidate) == 1, "unexpected cache.py contents"
cache_path.write_text(cache_source.replace(old_invalidate, new_invalidate))

store_path = Path("settings_store.py")
store_source = store_path.read_text()
old_set = '''    def set(self, key: str, value: object) -> None:
        """Change the value of ``key``."""
        self.source.write(key, value)'''
new_set = '''    def set(self, key: str, value: object) -> None:
        """Change the value of ``key`` and drop the now-stale cached copy."""
        self.source.write(key, value)
        self._cache.invalidate(key)'''
assert store_source.count(old_set) == 1, "unexpected settings_store.py contents"
store_path.write_text(store_source.replace(old_set, new_set))
PY

python3 -m pytest tests -q
