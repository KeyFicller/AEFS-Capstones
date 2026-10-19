"""进程内去重。单测会调 reset。"""

_seen = set()


def seen(key):
    return key in _seen


def mark(key):
    _seen.add(key)


def reset():
    _seen.clear()
