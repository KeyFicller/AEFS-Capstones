from terminal_coding_agent.viz import compose_mermaid

_PARENT = """---
config:
  flowchart:
    curve: linear
---
graph TD;
\t__start__([<p>__start__</p>]):::first
\trun_agent(run_agent)
\tsummary(summary)
\t__end__([<p>__end__</p>]):::last
\t__start__ --> run_agent;
\trun_agent --> summary;
\tclassDef default fill:#f2f0ff
"""

_INNER = """---
config:
  flowchart:
    curve: linear
---
graph TD;
\t__start__([<p>__start__</p>]):::first
\tmodel(model)
\ttools(tools)
\tM\\2ebefore(M.before)
\t__end__([<p>__end__</p>]):::last
\tmodel --> M\\2ebefore;
\tM\\2ebefore --> tools;
\ttools -.-> model;
\tclassDef default fill:#f2f0ff
"""


def test_the_node_box_becomes_a_subgraph() -> None:
    out = compose_mermaid(_PARENT, _INNER, "run_agent")

    assert 'subgraph run_agent["run_agent"]' in out
    assert "\n  end\n" in out
    assert "\trun_agent(run_agent)" not in out, "the plain box must be replaced, not kept"


def test_parent_edges_still_point_at_the_node() -> None:
    out = compose_mermaid(_PARENT, _INNER, "run_agent")

    assert "\t__start__ --> run_agent;" in out
    assert "\trun_agent --> summary;" in out


def test_inner_ids_are_prefixed_so_they_cannot_collide() -> None:
    """Both graphs own `__start__`; without a prefix Mermaid would merge the two."""
    out = compose_mermaid(_PARENT, _INNER, "run_agent")

    assert "agent____start__" in out
    assert "agent__model(model)" in out
    assert "agent__model --> agent__M\\2ebefore;" in out
    assert "agent__tools -.-> agent__model;" in out


def test_inner_chrome_is_dropped() -> None:
    out = compose_mermaid(_PARENT, _INNER, "run_agent")

    assert out.count("graph TD;") == 1, "the inner `graph TD;` header must not nest"
    assert out.count("classDef default") == 1


def test_an_unknown_node_leaves_the_parent_untouched() -> None:
    assert compose_mermaid(_PARENT, _INNER, "nope") == _PARENT
