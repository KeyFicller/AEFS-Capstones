"""The graph's job is to *terminate*, so these tests are about termination.

A run must never outlive ``max_rounds``, must never hand an empty page pool to the
answerer, and must say on the way out why it stopped -- or say nothing, which means it
finished normally. Each of those is pinned below, including the round-count trap: with
the counter bumped in both ``assess`` and ``verify``, a run that always asks for more
reaches ``max_rounds + 1`` before anyone notices.
"""

from pathlib import Path

from langchain_core.documents import Document
from multimodal_doc_qa.budget import Budget
from multimodal_doc_qa.config import Settings
from multimodal_doc_qa.graph import GraphDeps, build_graph, initial_state
from multimodal_doc_qa.schemas import Answer, Citation
from PIL import Image


class _Retriever:
    """Returns ``page_ids`` for every query, optionally nothing (a miss)."""

    def __init__(self, *pages: str) -> None:
        self.pages = pages
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        return [Document(page_content=p, metadata={"page_id": p, "score": 1.0}) for p in self.pages]


class _StubModel:
    """Answers whichever node calls it; the node reads only its own attribute."""

    def __init__(
        self,
        subqueries: tuple[str, ...] = ("sub",),
        followups: tuple[str, ...] = (),
        unsupported: tuple[str, ...] = (),
    ) -> None:
        self.subqueries = list(subqueries)
        self.followups = list(followups)
        self.unsupported = list(unsupported)
        self.calls: list[list[dict]] = []

    def invoke(self, messages: list[dict]) -> _StubModel:
        self.calls.append(messages)
        return self


class _FakeSynth:
    def __init__(self, cited: bool = True) -> None:
        self.cited = cited
        self.pools: list[list[str]] = []

    def synthesize(
        self, messages: object, page_ids: list[str], render_dir: Path, documents: object = None
    ) -> Answer:
        self.pools.append(list(page_ids))
        citations = [Citation(doc_id="doc000", page=0)] if self.cited else []
        return Answer(text="16.8%", citations=citations)


def _pages(tmp_path: Path, *page_ids: str) -> None:
    """assess and verify are shown page images, so the pages must exist on disk."""
    for pid in page_ids:
        path = tmp_path / f"{pid}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 3), "white").save(path)


def _run(tmp_path: Path, deps: GraphDeps, **settings: object) -> dict:
    return build_graph(deps, Settings(**settings)).invoke(
        initial_state("what was the EMEA margin?")
    )


def _deps(
    tmp_path: Path,
    retriever: object,
    model: _StubModel,
    synth: _FakeSynth | None = None,
    budget: Budget | None = None,
) -> GraphDeps:
    return GraphDeps(
        retriever=retriever,
        planner_model=model,
        assessor_model=model,
        verifier_model=model,
        synth=synth or _FakeSynth(),
        render_dir=tmp_path,
        budget=budget,
    )


# ------------------------------------------------------------------ normal finish


def test_a_single_hop_ask_retrieves_once_and_stops(tmp_path: Path) -> None:
    _pages(tmp_path, "doc000/p000")

    out = _run(tmp_path, _deps(tmp_path, _Retriever("doc000/p000"), _StubModel()))

    assert out["rounds"] == 1
    assert out["page_ids"] == ["doc000/p000"]


def test_a_normal_finish_leaves_stop_reason_unset(tmp_path: Path) -> None:
    """``stop_reason`` says why a run was cut short; a completed run has nothing to say."""
    _pages(tmp_path, "doc000/p000")

    out = _run(tmp_path, _deps(tmp_path, _Retriever("doc000/p000"), _StubModel()))

    assert out["stop_reason"] == ""


def test_the_answer_reaches_state_as_a_round_trippable_dict(tmp_path: Path) -> None:
    _pages(tmp_path, "doc000/p000")

    out = _run(tmp_path, _deps(tmp_path, _Retriever("doc000/p000"), _StubModel()))

    assert Answer.model_validate(out["answer"]).text == "16.8%"


def test_retrieval_is_queried_with_every_subquery(tmp_path: Path) -> None:
    _pages(tmp_path, "doc000/p000")
    retriever = _Retriever("doc000/p000")
    model = _StubModel(subqueries=("margin", "coverage"))

    _run(tmp_path, _deps(tmp_path, retriever, model))

    assert retriever.queries == ["margin", "coverage"]


# ------------------------------------------------------------------ empty retrieval


def test_a_retriever_that_misses_is_routed_to_recover(tmp_path: Path) -> None:
    """An empty pool cannot be answered from, so the run must not reach the answerer."""
    synth = _FakeSynth()

    out = _run(tmp_path, _deps(tmp_path, _Retriever(), _StubModel(), synth))

    assert out["stop_reason"] == "recover_empty"
    assert synth.pools == []


def test_an_empty_pool_is_recovered_even_when_the_assessor_wants_more(
    tmp_path: Path,
) -> None:
    model = _StubModel(followups=("more",))

    out = _run(tmp_path, _deps(tmp_path, _Retriever(), model))

    assert out["stop_reason"] == "recover_empty"


# ----------------------------------------------------------------------- the bound


def test_rounds_never_exceed_max_even_when_the_loop_always_asks_for_more(
    tmp_path: Path,
) -> None:
    """The bound is the whole point of ``max_rounds``; off-by-one here means unbounded spend.

    Both ``assess`` and ``verify`` can push the loop back to ``retrieve``, so a counter
    bumped in each of them reaches ``max_rounds + 1`` before the guard on the next hop
    ever runs.
    """
    _pages(tmp_path, "doc000/p000")
    model = _StubModel(followups=("more",), unsupported=("nope",))

    out = _run(tmp_path, _deps(tmp_path, _Retriever("doc000/p000"), model), max_rounds=3)

    assert out["rounds"] <= 3
    assert out["stop_reason"] == "recover_exhausted"


def test_rounds_counts_retrievals_not_model_calls(tmp_path: Path) -> None:
    """One retrieval is one round, so a single-hop ask reports a single round."""
    _pages(tmp_path, "doc000/p000")

    out = _run(tmp_path, _deps(tmp_path, _Retriever("doc000/p000"), _StubModel()), max_rounds=3)

    assert out["rounds"] == 1


def test_the_bound_holds_at_the_smallest_possible_max(tmp_path: Path) -> None:
    _pages(tmp_path, "doc000/p000")
    model = _StubModel(followups=("more",), unsupported=("nope",))

    out = _run(tmp_path, _deps(tmp_path, _Retriever("doc000/p000"), model), max_rounds=1)

    assert out["rounds"] <= 1


# -------------------------------------------------------------- looping back around


def test_a_verify_pushback_costs_another_retrieval_round(tmp_path: Path) -> None:
    _pages(tmp_path, "doc000/p000")
    model = _StubModel(unsupported=("unsupported claim",))

    out = _run(tmp_path, _deps(tmp_path, _Retriever("doc000/p000"), model), max_rounds=2)

    assert out["rounds"] == 2


def test_the_pool_accumulates_across_rounds_without_duplicates(tmp_path: Path) -> None:
    """Re-fetching a page already in the pool must not grow it: the pool is the budget."""
    _pages(tmp_path, "doc000/p000")
    model = _StubModel(unsupported=("unsupported claim",))

    out = _run(tmp_path, _deps(tmp_path, _Retriever("doc000/p000"), model), max_rounds=2)

    assert out["page_ids"] == ["doc000/p000"]


def test_an_assess_followup_costs_another_retrieval_round(tmp_path: Path) -> None:
    _pages(tmp_path, "doc000/p000")
    model = _StubModel(followups=("more",))

    out = _run(tmp_path, _deps(tmp_path, _Retriever("doc000/p000"), model), max_rounds=2)

    assert out["rounds"] == 2


def test_recovery_tells_exhaustion_apart_from_an_empty_pool(tmp_path: Path) -> None:
    """Eval has to tell these apart: one is a hard question, the other is a broken index."""
    _pages(tmp_path, "doc000/p000")
    model = _StubModel(followups=("more",))

    exhausted = _run(tmp_path, _deps(tmp_path, _Retriever("doc000/p000"), model), max_rounds=1)
    empty = _run(tmp_path, _deps(tmp_path, _Retriever(), model))

    assert exhausted["stop_reason"] != empty["stop_reason"]


def test_the_answerer_is_never_shown_an_empty_page_pool(tmp_path: Path) -> None:
    """Guards the empty-pool path against a later refactor rerouting it to synthesize."""
    _pages(tmp_path, "doc000/p000")
    synth = _FakeSynth()

    _run(tmp_path, _deps(tmp_path, _Retriever(), _StubModel(), synth))

    assert "" not in synth.pools
    assert [] not in synth.pools


# ------------------------------------------------------------------ budget


def _capped(**caps: object) -> Budget:
    """A budget whose other two caps are off, so one cap can be tested in isolation."""
    off: dict = {"max_calls": None, "max_tokens": None, "max_seconds": None}
    return Budget(**{**off, **caps})  # type: ignore[arg-type]


def test_a_spent_budget_never_reaches_the_first_model_call(tmp_path: Path) -> None:
    """The plan is the first thing that costs money, so it is the first thing to refuse."""
    _pages(tmp_path, "doc000/p000")
    retriever = _Retriever("doc000/p000")
    model = _StubModel()

    out = _run(
        tmp_path,
        _deps(tmp_path, retriever, model, budget=_capped(max_calls=0)),
    )

    assert out["stop_reason"] == "budget_exhausted"
    assert model.calls == []
    assert retriever.queries == []


def test_a_spent_token_budget_stops_the_run(tmp_path: Path) -> None:
    """Tokens are reported by the provider, not counted by the graph, so the gate has to
    read the tally rather than derive it."""
    _pages(tmp_path, "doc000/p000")
    model = _StubModel()
    budget = _capped(max_tokens=100)
    budget.tokens = 100

    out = _run(tmp_path, _deps(tmp_path, _Retriever("doc000/p000"), model, budget=budget))

    assert out["stop_reason"] == "budget_exhausted"
    assert model.calls == []


def test_no_model_call_happens_after_the_call_cap_is_spent(tmp_path: Path) -> None:
    """``budget.calls`` is reserved at the node boundary, so it must bound real calls."""
    _pages(tmp_path, "doc000/p000")
    model = _StubModel()
    budget = _capped(max_calls=2)

    out = _run(tmp_path, _deps(tmp_path, _Retriever("doc000/p000"), model, budget=budget))

    assert len(model.calls) == 2
    assert budget.calls == 2
    assert out["stop_reason"] == "budget_exhausted"


def test_exhaustion_keeps_the_answer_it_already_paid_for(tmp_path: Path) -> None:
    """A partial answer plus a reason beats nothing plus a reason: three calls buy
    ``plan`` + ``assess`` + ``synthesize``, and the fourth thing the run wants is ``verify``."""
    _pages(tmp_path, "doc000/p000")

    out = _run(
        tmp_path,
        _deps(tmp_path, _Retriever("doc000/p000"), _StubModel(), budget=_capped(max_calls=3)),
    )

    assert out["stop_reason"] == "budget_exhausted"
    assert Answer.model_validate(out["answer"]).text == "16.8%"


def test_budget_exhaustion_is_not_reported_as_a_retrieval_failure(tmp_path: Path) -> None:
    """Eval has to tell a spent budget apart from a broken index, as with the two recover
    reasons -- ``recover`` is about the corpus, ``budget_exhausted`` is about the wallet."""
    _pages(tmp_path, "doc000/p000")

    out = _run(
        tmp_path,
        _deps(tmp_path, _Retriever("doc000/p000"), _StubModel(), budget=_capped(max_calls=0)),
    )

    assert out["stop_reason"] not in {"recover_empty", "recover_exhausted"}


def test_the_worst_case_ask_costs_the_default_call_cap(tmp_path: Path) -> None:
    """The number ``max_ask_calls`` exists to cover, measured rather than reasoned about.

    Taking the ``verify`` loop-back every round runs ``plan`` then three full rounds of
    ``assess`` + ``synthesize`` + ``verify``: ``1 + 3 * 3`` = 10. The ``assess`` loop-back
    is cheaper (6) because once it stops looping the run synthesizes once. Both are pinned
    so that a change to the routing shows up as a call-count change, which is what the cap
    and the cost estimate depend on.
    """
    _pages(tmp_path, "doc000/p000")

    looping_on_verify = _capped()
    _run(
        tmp_path,
        _deps(
            tmp_path,
            _Retriever("doc000/p000"),
            _StubModel(unsupported=("missing",)),
            budget=looping_on_verify,
        ),
        max_rounds=3,
    )

    looping_on_assess = _capped()
    _run(
        tmp_path,
        _deps(
            tmp_path,
            _Retriever("doc000/p000"),
            _StubModel(followups=("more",)),
            budget=looping_on_assess,
        ),
        max_rounds=3,
    )

    assert looping_on_verify.calls == 10
    assert looping_on_assess.calls == 6


def test_an_uncapped_run_finishes_normally(tmp_path: Path) -> None:
    """The gate must not turn a healthy run into a truncated one.

    A single-hop ask costs four calls -- ``plan`` + ``assess`` + ``synthesize`` + ``verify``
    -- which is the floor the call cap has to clear before it can be useful.
    """
    _pages(tmp_path, "doc000/p000")
    budget = _capped()

    out = _run(tmp_path, _deps(tmp_path, _Retriever("doc000/p000"), _StubModel(), budget=budget))

    assert out["stop_reason"] == ""
    assert budget.calls == 4


# ------------------------------------------------------------ injected retriever shape


class _MetadataLessRetriever:
    """A retriever whose documents carry no ``page_id``. Retriever swapping is a first-class case."""

    def invoke(self, query: str) -> list[Document]:
        return [Document(page_content="page", metadata={"score": 1.0})]


def test_a_document_without_a_page_id_is_skipped_not_fatal(tmp_path: Path) -> None:
    """The retriever is injected ``Any``; a missing key must degrade, not raise ``KeyError``."""
    out = _run(tmp_path, _deps(tmp_path, _MetadataLessRetriever(), _StubModel()))

    assert out["page_ids"] == []
    assert out["stop_reason"] == "recover_empty"


def test_a_zero_round_budget_never_retrieves(tmp_path: Path) -> None:
    """The entry hop bypasses the assess/verify guards, so the bound must hold in ``retrieve``."""
    _pages(tmp_path, "doc000/p000")
    retriever = _Retriever("doc000/p000")

    out = _run(tmp_path, _deps(tmp_path, retriever, _StubModel()), max_rounds=0)

    assert out["rounds"] == 0
    assert retriever.queries == []


def test_a_new_answer_invalidates_the_previous_verification(tmp_path: Path) -> None:
    """If ``verify`` is skipped on budget, the fresh answer must not pair with a stale verdict."""
    _pages(tmp_path, "doc000/p000")
    model = _StubModel(unsupported=("nope",))

    out = _run(
        tmp_path,
        _deps(tmp_path, _Retriever("doc000/p000"), model, budget=_capped(max_calls=6)),
        max_rounds=2,
    )

    assert out["stop_reason"] == "budget_exhausted"
    assert out["unsupported"] == []
