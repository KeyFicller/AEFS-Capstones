"""可见回归：同主题的两封信都得发出去。"""

from dedupe import reset
from dispatcher import dispatch


def test_same_subject_reaches_both_users():
    reset()
    calls = []

    def sender(channel, job):
        calls.append((job["user_id"], channel))

    def job(user_id):
        return {
            "id": user_id,
            "user_id": user_id,
            "subject": "hi",
            "channels": ["email"],
        }

    dispatch(job("a"), "10:00", sender)
    dispatch(job("b"), "10:00", sender)

    assert calls == [("a", "email"), ("b", "email")]
