from app import App


def test_first_switch_reports_the_new_mode():
    app = App({"mode": "prod"})
    assert app.switch_to("dev") == "running in dev mode"
