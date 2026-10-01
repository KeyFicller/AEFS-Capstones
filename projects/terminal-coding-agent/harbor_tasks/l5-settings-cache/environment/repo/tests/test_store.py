from settings_store import SettingsStore


def test_default_value_is_returned():
    store = SettingsStore({"mode": "prod"})
    assert store.get("mode") == "prod"


def test_set_value_on_a_fresh_key_is_visible():
    store = SettingsStore()
    store.set("mode", "dev")
    assert store.get("mode") == "dev"


def test_missing_key_falls_back_to_the_default():
    store = SettingsStore()
    assert store.get("mode", "prod") == "prod"
