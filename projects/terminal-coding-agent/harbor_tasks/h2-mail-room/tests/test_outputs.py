"""Hidden checks for h2-mail-room. Never copied into the agent's repo."""

from dedupe import reset
from dispatcher import dispatch


def _job(**overrides):
    job = {
        "id": "n1",
        "user_id": "u1",
        "subject": "hi",
        "channels": ["email", "sms"],
    }
    job.update(overrides)
    return job


def test_quiet_hours_wrap_past_midnight():
    reset()
    calls = []

    def sender(channel, job):
        calls.append(channel)

    first = dispatch(_job(), "23:30", sender)
    assert first["sent"] == ["sms"]
    assert first["skipped"] == ["email"]
    assert calls == ["sms"]

    still = dispatch(_job(), "06:59", sender)
    assert still["sent"] == []
    assert still["skipped"] == ["email"]
    assert calls == ["sms"]

    opened = dispatch(_job(), "07:00", sender)
    assert opened["sent"] == ["email"]
    assert opened["skipped"] == []
    assert calls == ["sms", "email"]


def test_twenty_two_hundred_is_quiet_and_sms_is_not_resent():
    reset()
    calls = []

    def sender(channel, job):
        calls.append((channel, job["id"]))

    at_open = dispatch(_job(channels=["sms"]), "21:59", sender)
    at_quiet = dispatch(_job(channels=["sms"]), "22:00", sender)

    assert at_open["sent"] == ["sms"]
    assert at_quiet["sent"] == []
    assert calls == [("sms", "n1")]


def test_failed_channel_is_retried_without_resending_success():
    reset()
    calls = []
    broken = {"email"}

    def sender(channel, job):
        calls.append(channel)
        if channel in broken:
            raise RuntimeError("vendor down")

    first = dispatch(_job(), "10:00", sender)
    assert first["sent"] == ["sms"]
    assert first["failed"] == ["email"]

    broken.clear()
    second = dispatch(_job(), "10:00", sender)
    assert second["sent"] == ["email"]
    assert second["failed"] == []
    assert calls == ["email", "sms", "email"]


def test_same_user_same_id_does_not_double_send():
    reset()
    calls = []

    def sender(channel, job):
        calls.append(channel)

    dispatch(_job(channels=["email"]), "10:00", sender)
    again = dispatch(_job(channels=["email"]), "11:00", sender)

    assert again["sent"] == []
    assert calls == ["email"]
