"""Application facade over the settings store."""

from settings_store import SettingsStore


class App:
    """Turns the configured mode into the label shown to users."""

    def __init__(self, defaults: dict[str, object] | None = None) -> None:
        self.settings = SettingsStore(defaults)

    def label(self) -> str:
        """Human label for the current mode."""
        return f"running in {self.settings.get('mode')} mode"

    def switch_to(self, mode: str) -> str:
        """Change modes and report the new label."""
        self.settings.set("mode", mode)
        return self.label()
