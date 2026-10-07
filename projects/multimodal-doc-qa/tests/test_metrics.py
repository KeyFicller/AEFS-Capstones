"""Retrieval and evidence metrics, and the runner's results-file contract."""

import json
import math
from contextlib import suppress
from pathlib import Path

import pytest
from multimodal_doc_qa.eval.metrics import (
    bbox_hit_rate,
    iou_at_threshold,
    ndcg_at_k,
    pool_recall,
    recall_at_k,
)
from multimodal_doc_qa.eval.run import RESULTS_PATH, QuestionRun, _percentile, run_eval
from multimodal_doc_qa.schemas import Answer, BBox, Citation, Question


def _citation(page: int, bbox: BBox | None = None, doc_id: str = "d") -> Citation:
    return Citation(doc_id=doc_id, page=page, bbox=bbox)


def _box(x0: float, y0: float, x1: float, y1: float) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


# ------------------------------------------------------------------ ndcg_at_k


def test_ndcg_is_one_when_the_relevant_page_is_ranked_first() -> None:
    assert ndcg_at_k(["a", "b"], {"a"}, k=5) == 1.0


def test_ndcg_is_zero_when_no_ranked_page_is_relevant() -> None:
    assert ndcg_at_k(["x", "y"], {"a"}, k=5) == 0.0


def test_ndcg_lowers_as_the_relevant_page_sinks() -> None:
    relevant = {"a"}

    first = ndcg_at_k(["a", "b", "c"], relevant, k=5)
    third = ndcg_at_k(["b", "c", "a"], relevant, k=5)

    assert first == 1.0
    assert 0.0 < third < first


def test_ndcg_ignores_pages_beyond_k() -> None:
    """``k`` is the window; a hit at rank k+1 must not count for it."""
    assert ndcg_at_k(["b", "c", "a"], {"a"}, k=2) == 0.0


def test_ndcg_is_zero_when_nothing_is_relevant() -> None:
    """An empty ideal gain has no ratio; reporting 1.0 would call it a perfect retrieval."""
    assert ndcg_at_k(["a"], set(), k=5) == 0.0


def test_ndcg_over_an_empty_ranking_is_zero() -> None:
    assert ndcg_at_k([], {"a"}, k=5) == 0.0


def test_a_second_relevant_page_cannot_score_above_one() -> None:
    """Two gold pages, both retrieved: the ideal gain is two hits, so this is exactly 1.0."""
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=5) == 1.0


def test_a_page_retrieved_twice_is_counted_once() -> None:
    """The regression the OCR arm found: nDCG@5 came out at 1.1632.

    The text retriever returns several chunks of one page, so a relevant page was scored once
    per chunk while the ideal gain was computed from a set of one. nDCG is bounded by 1, so
    anything above it is a bug in the metric rather than a good retrieval.
    """
    assert ndcg_at_k(["a", "a", "a"], {"a"}, k=5) == 1.0


def test_ndcg_never_exceeds_one_when_ids_repeat() -> None:
    ranked = ["a", "a", "b", "b", "a", "c"]

    assert ndcg_at_k(ranked, {"a", "b"}, k=5) <= 1.0


def test_deduplication_happens_before_k_is_applied() -> None:
    """``k`` counts pages: a page seen three times must not consume three of the ``k`` slots.

    ``b`` is only inside a ``k=2`` window once the repeats of ``a`` are collapsed, and then
    it sits at rank 2, so its gain is ``1 / log2(3)``. Without collapsing it, ``b`` would be
    outside the window and the score would be 0.
    """
    assert ndcg_at_k(["a", "a", "a", "b"], {"b"}, k=2) == pytest.approx(1 / math.log2(3))


# ------------------------------------------------------------------ bbox_hit_rate


def test_bbox_hit_rate_requires_containment() -> None:
    gold = [_citation(1, _box(0.2, 0.2, 0.3, 0.3))]

    assert bbox_hit_rate([_citation(1, _box(0.1, 0.1, 0.4, 0.4))], gold) == 1.0
    assert bbox_hit_rate([_citation(1, _box(0.0, 0.0, 0.1, 0.1))], gold) == 0.0


def test_bbox_hit_rate_requires_the_same_page() -> None:
    gold = [_citation(1, _box(0.2, 0.2, 0.3, 0.3))]

    assert bbox_hit_rate([_citation(2, _box(0.1, 0.1, 0.4, 0.4))], gold) == 0.0


def test_bbox_hit_rate_requires_the_same_document() -> None:
    """Page 1 exists in every document, so a page match alone would be a false hit."""
    gold = [_citation(1, _box(0.2, 0.2, 0.3, 0.3), doc_id="doc000")]

    assert bbox_hit_rate([_citation(1, _box(0.1, 0.1, 0.4, 0.4), doc_id="doc001")], gold) == 0.0


def test_bbox_hit_rate_is_the_fraction_of_gold_covered() -> None:
    gold = [_citation(1, _box(0.2, 0.2, 0.3, 0.3)), _citation(2, _box(0.5, 0.5, 0.6, 0.6))]
    citations = [_citation(1, _box(0.1, 0.1, 0.4, 0.4))]

    assert bbox_hit_rate(citations, gold) == 0.5


def test_page_level_gold_counts_for_any_citation_on_that_page() -> None:
    gold = [_citation(3)]

    assert bbox_hit_rate([_citation(3, _box(0.0, 0.0, 0.1, 0.1))], gold) == 1.0


def test_a_gold_box_without_a_citation_box_is_a_miss() -> None:
    """The claim is not anchored, so the evidence region was not actually located."""
    gold = [_citation(1, _box(0.2, 0.2, 0.3, 0.3))]

    assert bbox_hit_rate([_citation(1)], gold) == 0.0


def test_bbox_hit_rate_over_an_answer_with_no_citations_is_zero() -> None:
    assert bbox_hit_rate([], [_citation(1, _box(0.2, 0.2, 0.3, 0.3))]) == 0.0


# ------------------------------------------------------------------ iou_at_threshold


def test_iou_is_one_for_identical_boxes_and_zero_without_overlap() -> None:
    box = _box(0.2, 0.2, 0.4, 0.4)

    assert box.iou(box) == pytest.approx(1.0)
    assert box.iou(_box(0.6, 0.6, 0.8, 0.8)) == 0.0


def test_iou_is_the_intersection_over_the_union() -> None:
    # Two 0.2 x 0.2 boxes overlapping over 0.1 x 0.2: 0.02 / (0.04 + 0.04 - 0.02).
    assert _box(0.0, 0.0, 0.2, 0.2).iou(_box(0.1, 0.0, 0.3, 0.2)) == pytest.approx(0.02 / 0.06)


def test_iou_at_threshold_needs_the_same_document_and_page() -> None:
    gold = [_citation(1, _box(0.2, 0.2, 0.4, 0.4), doc_id="doc000")]

    assert iou_at_threshold([_citation(2, _box(0.2, 0.2, 0.4, 0.4), doc_id="doc000")], gold) == 0.0
    assert iou_at_threshold([_citation(1, _box(0.2, 0.2, 0.4, 0.4), doc_id="doc001")], gold) == 0.0
    assert iou_at_threshold([_citation(1, _box(0.2, 0.2, 0.4, 0.4), doc_id="doc000")], gold) == 1.0


def test_iou_at_threshold_counts_a_page_level_gold_as_hit() -> None:
    assert iou_at_threshold([_citation(3, _box(0.0, 0.0, 0.5, 0.5))], [_citation(3)]) == 1.0


def test_iou_at_threshold_over_no_gold_is_zero() -> None:
    assert iou_at_threshold([_citation(1, _box(0.0, 0.0, 0.5, 0.5))], []) == 0.0


def test_iou_at_threshold_is_the_fraction_of_gold_matched() -> None:
    gold = [_citation(1, _box(0.2, 0.2, 0.4, 0.4)), _citation(2, _box(0.5, 0.5, 0.7, 0.7))]
    citations = [_citation(1, _box(0.2, 0.2, 0.4, 0.4))]

    assert iou_at_threshold(citations, gold) == 0.5


def test_the_strict_and_iou_metrics_rank_real_boxes_the_opposite_way() -> None:
    """The regression that made IoU the headline metric, on boxes measured from a real run.

    ``doc000-p0``: the model drew a box 0.67 of the page wide around a gold line 0.24 wide,
    so it *contains* it and strict containment scores a hit. ``doc001-p2``: the model's box
    nearly coincides with the gold one but its right edge falls short by 0.005 of the page
    width, so strict containment scores a total miss. Read one after the other, containment
    says the sloppy box was right and the accurate one was wrong.
    """
    wide_gold = _citation(0, _box(0.34637, 0.08096, 0.58303, 0.10376))
    wide_model = _citation(0, _box(0.11, 0.078, 0.78, 0.178))

    tight_gold = _citation(2, _box(0.330242, 0.218111, 0.694758, 0.252889))
    tight_model = _citation(2, _box(0.33, 0.21, 0.69, 0.26))

    assert bbox_hit_rate([wide_model], [wide_gold]) == 1.0
    assert iou_at_threshold([wide_model], [wide_gold]) == 0.0

    assert bbox_hit_rate([tight_model], [tight_gold]) == 0.0
    assert iou_at_threshold([tight_model], [tight_gold]) == 1.0


def test_the_threshold_is_a_real_parameter() -> None:
    """0.4 of the page area in common is not enough at 0.5, and the caller controls that."""
    gold = [_citation(1, _box(0.0, 0.0, 0.2, 0.2))]
    citations = [_citation(1, _box(0.1, 0.1, 0.3, 0.3))]  # IoU 0.01/0.07

    assert iou_at_threshold(citations, gold, threshold=0.5) == 0.0
    assert iou_at_threshold(citations, gold, threshold=0.1) == 1.0


def test_a_threshold_outside_the_unit_interval_is_rejected() -> None:
    """A threshold <= 0 counts a zero-overlap box as a hit and silently inflates the metric."""
    gold = [_citation(1, _box(0.0, 0.0, 0.2, 0.2))]
    citations = [_citation(1, _box(0.8, 0.8, 0.9, 0.9))]

    assert iou_at_threshold(citations, gold, threshold=1.0) == 0.0
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            iou_at_threshold(citations, gold, threshold=bad)


def test_the_percentile_uses_the_nearest_rank_not_bankers_rounding() -> None:
    """Nearest rank is ``ceil(fraction * N)``; ``round`` shifts p50/p95 with N's parity."""
    five = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert _percentile(five, 0.50) == 3.0  # round(2.5) would pick the 2nd value
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.0
    assert _percentile(five, 0.95) == 5.0
    assert _percentile([], 0.5) == 0.0


# ------------------------------------------------------------------ runner


def _question(qid: str = "q1", page: int = 0) -> Question:
    return Question(
        qid=qid,
        text=f"what is on page {page}?",
        answer="16.8%",
        evidence=[_citation(page, _box(0.1, 0.1, 0.2, 0.2), doc_id="doc000")],
        hops=1,
    )


def _run(
    page: int = 0, stop_reason: str = "", cited: bool = True, answered: bool = True
) -> QuestionRun:
    """One question's outcome. ``answered=False`` is the graph refusing to answer at all."""
    from multimodal_doc_qa.schemas import page_id

    answer = None
    if answered:
        answer = Answer(
            text="16.8%",
            citations=[_citation(page, _box(0.0, 0.0, 0.5, 0.5), doc_id="doc000")] if cited else [],
        )
    return QuestionRun(
        answer=answer,
        ranked=[page_id("doc000", page)],
        pool=[page_id("doc000", page)],
        rounds=1,
        calls=4,
        tokens=100,
        stop_reason=stop_reason,
    )


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_results_do_not_land_in_the_source_tree() -> None:
    """The first version resolved ``parents[2]`` from inside the package: ``src/eval/``."""
    assert RESULTS_PATH.parent == RESULTS_PATH.parents[1] / "eval"
    assert "src" not in RESULTS_PATH.parts


def test_the_runner_writes_a_run_line_a_question_line_and_a_summary(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"

    run_eval([_question()], lambda q: _run(), out_path=out, mode="maxsim", k=5)

    kinds = [row["kind"] for row in _read(out)]
    assert kinds == ["run", "question", "summary"]


def test_the_run_line_carries_provenance(tmp_path: Path) -> None:
    """Without it a number in the file cannot be traced to a commit or a model."""
    out = tmp_path / "results.jsonl"

    run_eval([_question()], lambda q: _run(), out_path=out, settings=_settings(), mode="ocr", k=5)

    run_line = _read(out)[0]
    assert run_line["mode"] == "ocr"
    assert run_line["commit"] is not None
    assert run_line["date"]
    assert run_line["embedder_model"] == "vidore/colSmol-500M"
    assert run_line["ocr_embedder_model"] == "BAAI/bge-small-en-v1.5"
    assert run_line["answerer_model"] == "deepseek:deepseek-flash"
    assert run_line["n_questions"] == 1
    # An IoU number cannot be compared to anything without the threshold behind it.
    assert run_line["iou_threshold"] == 0.5


def test_the_threshold_a_run_used_is_the_one_recorded(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"

    run_eval(
        [_question()], lambda q: _run(), out_path=out, settings=_settings(), k=5, iou_threshold=0.03
    )

    rows = _read(out)
    assert rows[0]["iou_threshold"] == 0.03
    # The fixture's IoU is 0.04, so it is a miss at the default 0.5 and a hit at 0.03.
    assert rows[1]["iou_at_threshold"] == 1.0


def test_each_question_line_carries_its_metrics_and_its_raw_material(tmp_path: Path) -> None:
    """The raw fields are what let a later metric be recomputed without re-running the model."""
    out = tmp_path / "results.jsonl"

    run_eval([_question()], lambda q: _run(), out_path=out, k=5)

    row = _read(out)[1]
    assert row["qid"] == "q1"
    assert row["ndcg_at_k"] == 1.0
    assert row["bbox_hit_rate"] == 1.0
    # The fixture cites a box covering half the page around a gold box of 1/20 its area, which
    # is exactly the case the two evidence metrics disagree on, so both are recorded.
    assert row["iou_at_threshold"] == 0.0
    assert row["answer"] == "16.8%"
    assert row["ranked"] == ["doc000/p000"]
    assert row["stop_reason"] == ""


def test_the_summary_aggregates_the_questions(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"

    summary = run_eval(
        [_question("q1"), _question("q2")],
        lambda q: _run(),
        out_path=out,
        k=5,
    )

    assert summary["n_done"] == 2
    assert summary["ndcg_at_k"] == 1.0
    assert summary["iou_at_threshold"] == 0.0
    assert summary["bbox_hit_rate"] == 1.0
    assert summary["tokens"] == 200
    assert summary["stop_reasons"] == {"": 2}


def test_a_missing_answer_scores_zero_rather_than_raising(tmp_path: Path) -> None:
    """A question the graph refused to answer still has to produce a row."""
    out = tmp_path / "results.jsonl"

    summary = run_eval(
        [_question()],
        lambda q: _run(answered=False, stop_reason="recover_empty"),
        out_path=out,
        k=5,
    )

    assert summary["bbox_hit_rate"] == 0.0
    assert _read(out)[1]["answer"] is None


def test_citations_without_a_box_are_still_recorded(tmp_path: Path) -> None:
    """A page-level claim is partially right and the row has to show what was claimed."""
    out = tmp_path / "results.jsonl"

    run_eval([_question()], lambda q: _run(cited=False), out_path=out, k=5)

    assert _read(out)[1]["bbox_hit_rate"] == 0.0


def test_the_run_line_is_written_before_any_question_is_answered(tmp_path: Path) -> None:
    """An interrupted run must still say what it belongs to; a header written last would not."""
    out = tmp_path / "results.jsonl"

    def explode(question: Question):
        raise RuntimeError("model died")

    with suppress(RuntimeError):
        run_eval([_question()], explode, out_path=out, settings=_settings(), k=5)

    rows = _read(out)
    assert [row["kind"] for row in rows] == ["run"]


def test_the_token_cap_stops_the_suite_and_says_where(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"

    summary = run_eval(
        [_question("q1"), _question("q2"), _question("q3")],
        lambda q: _run(),
        out_path=out,
        k=5,
        max_tokens=250,
    )

    assert summary["n_done"] == 3
    assert [row["qid"] for row in _read(out) if row["kind"] == "question"] == ["q1", "q2", "q3"]


def test_the_cap_leaves_the_already_written_rows_in_place(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"

    summary = run_eval(
        [_question("q1"), _question("q2"), _question("q3")],
        lambda q: _run(),
        out_path=out,
        k=5,
        max_tokens=150,
    )

    assert summary["n_done"] == 2
    assert summary["stopped_after"] == "q3"
    assert len([row for row in _read(out) if row["kind"] == "question"]) == 2


def test_two_runs_append_without_merging(tmp_path: Path) -> None:
    """Appending is right, as long as each run is delimited and identified."""
    out = tmp_path / "results.jsonl"

    run_eval(
        [_question()], lambda q: _run(), out_path=out, settings=_settings(), mode="maxsim", k=5
    )
    run_eval([_question()], lambda q: _run(), out_path=out, settings=_settings(), mode="ocr", k=5)

    rows = _read(out)
    assert [row["kind"] for row in rows] == [
        "run",
        "question",
        "summary",
        "run",
        "question",
        "summary",
    ]
    assert [row["mode"] for row in rows if row["kind"] == "run"] == ["maxsim", "ocr"]


def _settings():
    from multimodal_doc_qa.config import Settings

    return Settings()


# ------------------------------------------------------------------ recall_at_k


def test_recall_is_one_when_every_gold_page_is_in_the_top_k() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=5) == 1.0


def test_recall_is_the_fraction_of_gold_pages_found() -> None:
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=5) == 0.5


def test_recall_is_zero_when_no_gold_page_is_ranked() -> None:
    assert recall_at_k(["x", "y"], {"a"}, k=5) == 0.0


def test_recall_ignores_pages_beyond_k() -> None:
    """``k`` is the window; a gold page at rank k+1 must not count."""
    assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_recall_counts_a_repeated_page_once() -> None:
    """The text arms emit several chunks of one page; each chunk is the same page."""
    assert recall_at_k(["a", "a", "a"], {"a", "b"}, k=5) == 0.5


def test_recall_is_zero_for_empty_gold() -> None:
    assert recall_at_k(["a"], set(), k=5) == 0.0


# ------------------------------------------------------------------ pool_recall


def test_pool_recall_is_order_independent() -> None:
    assert pool_recall(["x", "b", "a"], {"a", "b"}) == 1.0


def test_pool_recall_counts_a_repeated_page_once() -> None:
    assert pool_recall(["a", "a"], {"a", "b"}) == 0.5


def test_pool_recall_is_zero_for_empty_gold() -> None:
    assert pool_recall(["a"], set()) == 0.0


# ------------------------------------------------------------------ runner recalls


def test_each_question_line_carries_both_recalls(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"

    run_eval([_question()], lambda q: _run(), out_path=out, k=5)

    row = _read(out)[1]
    assert row["recall_at_k"] == 1.0
    assert row["pool_recall"] == 1.0


def test_the_summary_averages_both_recalls(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"

    run_eval([_question()], lambda q: _run(), out_path=out, k=5)

    summary = _read(out)[-1]
    assert summary["recall_at_k"] == 1.0
    assert summary["pool_recall"] == 1.0
