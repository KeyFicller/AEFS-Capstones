"""Hidden checks for l5-settings-cache. Never copied into the agent's repo."""

from app import App
from settings_store import SettingsStore


def test_set_after_a_read_is_visible():
    store = SettingsStore({"mode": "prod"})
    assert store.get("mode") == "prod"
    store.set("mode", "dev")
    assert store.get("mode") == "dev"


def test_two_consecutive_sets_are_both_visible():
    store = SettingsStore()
    store.set("mode", "dev")
    assert store.get("mode") == "dev"
    store.set("mode", "staging")
    assert store.get("mode") == "staging"


def test_writes_reach_the_source():
    store = SettingsStore({"mode": "prod"})
    store.set("mode", "dev")
    assert store.source.read("mode") == "dev"


def test_repeated_reads_do_not_touch_the_slow_source():
    store = SettingsStore({"mode": "prod"})
    store.get("mode")
    store.get("mode")
    store.get("mode")
    assert store.source.reads == 1


def test_switch_to_reports_the_new_mode():
    app = App({"mode": "prod"})
    app.settings.get("mode")
    assert app.switch_to("dev") == "running in dev mode"


def test_switching_twice_reports_the_latest_mode():
    app = App({"mode": "prod"})
    assert app.switch_to("dev") == "running in dev mode"
    assert app.switch_to("staging") == "running in staging mode"
