"""Session stitching: chain events that are at most 30 minutes apart."""

GAP_MS = 30 * 60 * 1000


def stitch(events: list[dict]) -> dict[int, str]:
    """Map every event index to its session id."""
    order = sorted(
        range(len(events)),
        key=lambda index: (
            events[index]["tenant"],
            events[index]["user"],
            events[index]["ts_ms"],
            index,
        ),
    )

    assignment: dict[int, str] = {}
    current_key = None
    start = 0
    previous = 0

    for index in order:
        event = events[index]
        key = (event["tenant"], event["user"])
        if key != current_key or event["ts_ms"] - previous > GAP_MS:
            start = event["ts_ms"]
            current_key = key
        assignment[index] = f"{key[0]}:{key[1]}:{start}"
        previous = event["ts_ms"]

    return assignment
