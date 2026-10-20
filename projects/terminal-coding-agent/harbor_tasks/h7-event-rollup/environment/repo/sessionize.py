"""Session stitching: chain events that are at most 30 minutes apart."""

GAP_MS = 30 * 60 * 1000


def _find(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _union(parent: list[int], left: int, right: int) -> None:
    root_left, root_right = _find(parent, left), _find(parent, right)
    if root_left != root_right:
        parent[max(root_left, root_right)] = min(root_left, root_right)


def stitch(events: list[dict]) -> dict[int, str]:
    """Map every event index to its session id."""
    groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for index, event in enumerate(events):
        groups.setdefault((event["tenant"], event["user"]), []).append((event["ts_ms"], index))

    assignment: dict[int, str] = {}
    for (tenant, user), items in groups.items():
        items.sort()
        size = len(items)
        parent = list(range(size))

        # Every pair that is close enough belongs to the same chain.
        for left in range(size):
            for right in range(left + 1, size):
                if items[right][0] - items[left][0] <= GAP_MS:
                    _union(parent, left, right)

        for position, (_, index) in enumerate(items):
            root = _find(parent, position)
            assignment[index] = f"{tenant}:{user}:{items[root][0]}"

    return assignment
