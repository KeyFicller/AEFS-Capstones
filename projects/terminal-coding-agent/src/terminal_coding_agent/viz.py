"""Compose one Mermaid flowchart from a parent graph and a graph the parent calls.

`get_graph(xray=True)` only expands subgraphs that were registered as nodes at compile
time. This project's executor node builds its `create_agent` graph inside the node, so
the parent diagram cannot see it. `compose_mermaid` pastes that inner graph — rendered
from the real compiled agent — into the parent's node box instead.
"""

import re

_INNER_SKIP = ("---", "config:", "flowchart:", "curve:", "graph TD;", "classDef ")
_DECLARATION = re.compile(r"^(\s*)(\S+?)([\(\[].*)$")
_EDGE = re.compile(r"^(\s*)(\S+)(\s+(?:-->|-\.->)\s+)(\S+?)(;?)\s*$")


def _inner_lines(inner: str, prefix: str) -> list[str]:
    """Re-emit the inner graph's nodes and edges under a prefix, without its chrome."""
    lines: list[str] = []
    for raw in inner.splitlines():
        line = raw.strip()
        if not line or line.startswith(_INNER_SKIP):
            continue
        if edge := _EDGE.match(line):
            arrow = edge.group(3)
            lines.append(f"{prefix}{edge.group(2)}{arrow}{prefix}{edge.group(4)}{edge.group(5)}")
        elif declaration := _DECLARATION.match(line):
            lines.append(f"{prefix}{declaration.group(2)}{declaration.group(3)}")
    return lines


def compose_mermaid(parent: str, inner: str, node_id: str, *, prefix: str = "agent__") -> str:
    """Replace the parent's `node_id` box with a subgraph holding the inner graph.

    Ids are prefixed so the inner graph cannot collide with parent nodes (`__start__`
    especially). Anything the parser does not recognise is dropped, so a Mermaid format
    change degrades to a plain parent diagram rather than a corrupt one.
    """
    inner_block = [f"  subgraph {node_id}[\"{node_id}\"]", "    direction TB"]
    inner_block += [f"    {line}" for line in _inner_lines(inner, prefix)]
    inner_block.append("  end")

    out: list[str] = []
    replaced = False
    for raw in parent.splitlines():
        declaration = _DECLARATION.match(raw.strip())
        if declaration and declaration.group(2) == node_id and not replaced:
            out.extend(inner_block)
            replaced = True
            continue
        out.append(raw)
    if not replaced:
        return parent
    return "\n".join(out) + "\n"
