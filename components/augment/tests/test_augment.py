"""The component is model-agnostic and budget-agnostic, so both are stubs here.

Every subgraph test invokes the subgraph directly, with a hand-built ``AugmentState`` --
no host graph is involved, which is the point of owning the state.
"""

import pytest
from augment import (
    MULTIQUERY_N,
    TECHNIQUES,
    Queries,
    build_augment_graph,
    decompose,
    hyde,
    latest_human_text,
    merge_queries,
    multiquery,
    parse_techniques,
    rewrite,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class _Model:
    """One queued ``Queries`` per call, recording every prompt it was handed."""

    def __init__(self, *replies: tuple[str, ...]) -> None:
        self.replies = [list(reply) for reply in replies]
        self.prompts: list[list] = []
        self.bound: object = None

    def with_structured_output(self, schema: object) -> "_Model":
        self.bound = schema
        return self

    def invoke(self, messages: list) -> Queries:
        self.prompts.append(messages)
        return Queries(queries=self.replies.pop(0))


class _Ledger:
    """The whole budget contract: ``spend_call`` and ``exhausted``. Nothing else is required."""

    def __init__(self, max_calls: int | None = None) -> None:
        self.max_calls = max_calls
        self.calls = 0

    def spend_call(self) -> None:
        self.calls += 1

    def exhausted(self) -> bool:
        return self.max_calls is not None and self.calls >= self.max_calls


def _text(messages: list) -> str:
    return "\n".join(str(message.content) for message in messages)


def _augmented(
    techniques: tuple[str, ...],
    replies: tuple[tuple[str, ...], ...],
    *,
    budget: _Ledger | None = None,
    render: object = None,
) -> tuple[dict, _Model]:
    model = _Model(*replies)
    kwargs: dict = {
        "techniques": techniques,
        "budget": budget if budget is not None else _Ledger(),
    }
    if render is not None:
        kwargs["render"] = render
    graph = build_augment_graph(model, **kwargs)
    out = graph.invoke(
        {
            "messages": [HumanMessage(content="And who founded it?")],
            "queries": [],
            "stop_reason": "",
        }
    )
    return out, model


# --------------------------------------------------------------- parse_techniques


def test_the_default_spec_names_rewrite_only() -> None:
    assert parse_techniques("rewrite") == ("rewrite",)


def test_an_empty_spec_names_no_technique() -> None:
    """``MDQ_AUGMENT=`` is the ablation baseline: the bare turn and no extra call."""
    assert parse_techniques("") == ()


def test_the_written_order_does_not_survive() -> None:
    """The output order is fixed, so a config string cannot reorder the merge."""
    assert parse_techniques("hyde,multiquery,decompose,rewrite") == TECHNIQUES


def test_whitespace_and_repeats_are_ignored() -> None:
    assert parse_techniques(" rewrite , rewrite ") == ("rewrite",)


def test_an_unknown_technique_is_an_error_not_a_silent_skip() -> None:
    with pytest.raises(ValueError, match="hyede"):
        parse_techniques("rewrite,hyede")


# ----------------------------------------------------------------------- techniques


def test_the_model_is_bound_to_the_queries_schema_once() -> None:
    model = _Model(("base",))

    build_augment_graph(model, techniques=("rewrite",), budget=_Ledger())

    assert model.bound is Queries


def test_rewrite_returns_the_first_query_the_model_produced() -> None:
    model = _Model(("Who founded Gestalt psychology?",))

    assert rewrite(model, [HumanMessage(content="And who founded it?")]) == (
        "Who founded Gestalt psychology?"
    )


def test_rewrite_returns_an_empty_string_when_the_model_produced_nothing() -> None:
    """The caller owns the fallback to the bare turn; an empty string is how it learns."""
    assert rewrite(_Model(()), [HumanMessage(content="compare margins")]) == ""


def test_rewrite_is_shown_the_conversation() -> None:
    model = _Model(("Who founded Gestalt psychology?",))

    rewrite(
        model,
        [
            HumanMessage(content="Where was Gestalt psychology conceived?"),
            AIMessage(content="Germany."),
            HumanMessage(content="And who founded it?"),
        ],
    )

    seen = _text(model.prompts[0])
    assert "Gestalt psychology" in seen
    assert "And who founded it?" in seen


def test_rewrite_uses_the_render_it_was_given() -> None:
    """Trimming history is the host's policy; the component must not quietly bypass it."""
    model = _Model(("x",))

    def render(system: str, messages: list) -> list:
        return [SystemMessage(content=system), HumanMessage(content="trimmed")]

    rewrite(model, [HumanMessage(content="And who founded it?")], render)

    assert _text(model.prompts[0]).endswith("trimmed")


@pytest.mark.parametrize("expander", [decompose, multiquery, hyde])
def test_an_expander_returns_what_the_model_produced(expander: object) -> None:
    assert expander(_Model(("a", "b")), "the base query") == ["a", "b"]


@pytest.mark.parametrize("expander", [decompose, multiquery, hyde])
def test_an_expander_sees_only_the_base_query(expander: object) -> None:
    """A reworder or a hypothetical answer needs the query, not the conversation."""
    model = _Model(("a",))

    expander(model, "the base query")

    assert _text(model.prompts[0]).endswith("the base query")


def test_the_multiquery_prompt_asks_for_the_configured_count() -> None:
    model = _Model(("a",))

    multiquery(model, "q")

    assert str(MULTIQUERY_N) in str(model.prompts[0][0].content)


# -------------------------------------------------------------------------- helpers


def test_merge_queries_drops_blanks_and_repeats_and_keeps_the_first_spelling() -> None:
    assert merge_queries(["base"], [], ["", "  Base ", "variant"]) == ["base", "variant"]


def test_merge_queries_keeps_the_order_of_the_groups() -> None:
    assert merge_queries(["a"], ["b", "c"], ["b"]) == ["a", "b", "c"]


def test_latest_human_text_is_the_last_human_turn() -> None:
    messages = [
        HumanMessage(content="first"),
        AIMessage(content="reply"),
        HumanMessage(content="last"),
    ]

    assert latest_human_text(messages) == "last"


def test_latest_human_text_reads_text_blocks_and_ignores_the_rest() -> None:
    message = HumanMessage(
        content=[
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "data:"}},
        ]
    )

    assert latest_human_text([message]) == "hello"


def test_latest_human_text_without_a_human_turn_is_empty() -> None:
    assert latest_human_text([AIMessage(content="hi")]) == ""


# -------------------------------------------------------------------------- subgraph


def test_an_empty_spec_passes_the_bare_turn_through() -> None:
    out, model = _augmented((), ())

    assert out["queries"] == ["And who founded it?"]
    assert model.prompts == []


def test_rewrite_only_outputs_the_rewritten_query() -> None:
    out, model = _augmented(("rewrite",), (("Who founded Gestalt psychology?",),))

    assert out["queries"] == ["Who founded Gestalt psychology?"]
    assert len(model.prompts) == 1


def test_everything_on_concatenates_in_the_fixed_segment_order() -> None:
    out, model = _augmented(TECHNIQUES, (("base",), ("s1", "s2"), ("q1", "q2", "q3"), ("h",)))

    assert out["queries"] == ["base", "s1", "s2", "q1", "q2", "q3", "h"]
    assert len(model.prompts) == 4


def test_a_technique_outside_the_set_is_never_entered() -> None:
    out, model = _augmented(("rewrite", "multiquery"), (("base",), ("paraphrase",)))

    assert len(model.prompts) == 2, "decompose is not in the set and must not be entered"
    assert out["queries"] == ["base", "paraphrase"]


def test_each_expander_is_given_the_base_and_not_the_conversation() -> None:
    _, model = _augmented(TECHNIQUES, (("base",), ("s",), ("q",), ("h",)))

    assert _text(model.prompts[1]).endswith("base")
    assert "And who founded it?" not in _text(model.prompts[1])


def test_a_rewrite_that_produced_nothing_falls_back_to_the_bare_turn() -> None:
    out, model = _augmented(("rewrite", "multiquery"), ((), ("q",)))

    assert out["queries"] == ["And who founded it?", "q"]
    assert _text(model.prompts[1]).endswith("And who founded it?")


def test_an_expander_works_without_rewrite() -> None:
    """Ablation: ``multiquery`` alone expands the bare turn."""
    out, model = _augmented(("multiquery",), (("q1", "q2"),))

    assert out["queries"] == ["And who founded it?", "q1", "q2"]
    assert _text(model.prompts[0]).endswith("And who founded it?")


def test_an_empty_decompose_only_costs_its_segment() -> None:
    out, _ = _augmented(("rewrite", "decompose", "hyde"), (("base",), (), ("h",)))

    assert out["queries"] == ["base", "h"]


def test_finalize_dedupes_what_two_techniques_overlapped_on() -> None:
    out, _ = _augmented(("rewrite", "decompose"), (("base",), ("Base", "variant")))

    assert out["queries"] == ["base", "variant"]


def test_a_spent_budget_skips_the_remaining_techniques() -> None:
    """A technique that cannot afford its call is jumped over, not entered and failed."""
    out, model = _augmented(TECHNIQUES, (("base",),), budget=_Ledger(max_calls=1))

    assert len(model.prompts) == 1
    assert out["queries"] == ["base"]
    assert out["stop_reason"] == ""


def test_a_budget_spent_on_entry_ends_the_subgraph() -> None:
    out, model = _augmented(TECHNIQUES, (), budget=_Ledger(max_calls=0))

    assert out["stop_reason"] == "budget_exhausted"
    assert out["queries"] == []
    assert model.prompts == []
