"""Evidence overlay. Boxes are checked by coordinate, on a rectangle that is not square."""

import json
from pathlib import Path

from multimodal_doc_qa.schemas import Answer, BBox, Citation
from multimodal_doc_qa.ui.viewer import (
    CITED,
    GOLD,
    PAGE_LEVEL,
    append_turn,
    citation_lines,
    draw_citations,
    evidence_lines,
    load_runs,
    row_citations,
    runs_for_question,
)
from PIL import Image


def _citation(doc_id: str = "d", page: int = 0, bbox: BBox | None = None) -> Citation:
    return Citation(doc_id=doc_id, page=page, bbox=bbox)


def _box(x0: float, y0: float, x1: float, y1: float) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


# ------------------------------------------------------------------ drawing


def test_draw_citations_marks_the_box_on_the_page() -> None:
    image = Image.new("RGB", (100, 200), "white")

    out = draw_citations(image, [_citation(bbox=_box(0.1, 0.2, 0.3, 0.4))], doc_id="d", page=0)

    assert out.getpixel((10, 40)) != (255, 255, 255)


def test_the_box_is_not_transposed() -> None:
    """A square box cannot catch an x/y swap, so this one is twice as tall as it is wide.

    The bbox covers x 10..30 and y 40..80 on a 100x200 page, so the corner is at (10, 40).
    Transposed code would draw it at (40, 10), which is empty here.
    """
    image = Image.new("RGB", (100, 200), "white")

    out = draw_citations(image, [_citation(bbox=_box(0.1, 0.2, 0.3, 0.4))], doc_id="d", page=0)

    assert out.getpixel((10, 40)) != (255, 255, 255)
    assert out.getpixel((40, 10)) == (255, 255, 255)


def _coloured_pixels(image: Image.Image, color: tuple[int, int, int]) -> int:
    """Count pixels of exactly ``color``, without the deprecated ``getdata()``."""
    return image.tobytes().count(bytes(color))


def test_draw_citations_ignores_another_page() -> None:
    """Multi-hop answers cite several pages; drawing them all on one page would be a lie."""
    image = Image.new("RGB", (100, 200), "white")
    citations = [_citation(page=1, bbox=_box(0.1, 0.2, 0.3, 0.4))]

    out = draw_citations(image, citations, doc_id="d", page=0)

    assert out.tobytes() == image.tobytes()


def test_draw_citations_ignores_another_document() -> None:
    """Every document has a page 0, so matching on the page alone is a false hit."""
    image = Image.new("RGB", (100, 200), "white")
    citations = [_citation(doc_id="other", page=0, bbox=_box(0.1, 0.2, 0.3, 0.4))]

    out = draw_citations(image, citations, doc_id="d", page=0)

    assert out.tobytes() == image.tobytes()


def test_draw_citations_does_not_mutate_its_input() -> None:
    image = Image.new("RGB", (100, 200), "white")

    draw_citations(image, [_citation(bbox=_box(0.1, 0.2, 0.3, 0.4))], doc_id="d", page=0)

    assert image.tobytes() == bytes((255, 255, 255)) * (100 * 200)


def test_a_page_level_citation_is_a_border_not_a_claimed_region() -> None:
    """No bbox means no region was located, which is a different claim from a full-page box.

    It is drawn as a border in its own colour: the interior stays clear, so the picture does
    not assert a located region where the model claimed only a page.
    """
    image = Image.new("RGB", (100, 200), "white")

    out = draw_citations(image, [_citation(page=0, bbox=None)], doc_id="d", page=0)

    assert _coloured_pixels(out, PAGE_LEVEL) > 0
    assert out.getpixel((50, 100)) == (255, 255, 255)
    assert _coloured_pixels(out, CITED) == 0


def test_a_box_is_drawn_in_the_colour_it_is_given() -> None:
    image = Image.new("RGB", (100, 200), "white")

    gold = draw_citations(
        image, [_citation(bbox=_box(0.1, 0.2, 0.3, 0.4))], doc_id="d", page=0, color=GOLD
    )
    cited = draw_citations(
        image, [_citation(bbox=_box(0.1, 0.2, 0.3, 0.4))], doc_id="d", page=0, color=CITED
    )

    assert gold.getpixel((10, 40)) == GOLD
    assert cited.getpixel((10, 40)) == CITED


def test_the_outline_scales_with_the_page() -> None:
    """A 3px outline is invisible on a 2048px render and a slab on a thumbnail."""
    tiny = Image.new("RGB", (100, 200), "white")
    large = Image.new("RGB", (1000, 2000), "white")

    small_edge = _coloured_pixels(
        draw_citations(tiny, [_citation(bbox=_box(0.1, 0.2, 0.3, 0.4))], doc_id="d", page=0), CITED
    )
    large_edge = _coloured_pixels(
        draw_citations(large, [_citation(bbox=_box(0.1, 0.2, 0.3, 0.4))], doc_id="d", page=0),
        CITED,
    )

    # Ten times the page should cost far more than ten times the ink, because the outline
    # thickens as well as lengthens.
    assert large_edge > small_edge * 10


def test_a_box_on_the_page_edge_stays_inside_the_image() -> None:
    image = Image.new("RGB", (100, 200), "white")

    out = draw_citations(image, [_citation(bbox=_box(0.0, 0.0, 1.0, 1.0))], doc_id="d", page=0)

    assert out.size == (100, 200)


def test_no_citations_leaves_the_page_untouched() -> None:
    image = Image.new("RGB", (100, 200), "white")

    out = draw_citations(image, [], doc_id="d", page=0)

    assert out.tobytes() == image.tobytes()


# ------------------------------------------------------------------ evidence text


def test_citation_lines_mark_a_box_and_a_page_level_claim_differently() -> None:
    lines = citation_lines(
        [_citation(doc_id="d", page=0, bbox=_box(0.1, 0.1, 0.2, 0.2)), _citation(page=3)]
    )

    assert "- `d/p000` — box" in lines
    assert "- `d/p003` — page-level" in lines


def test_citation_lines_say_so_when_there_are_none() -> None:
    assert citation_lines([]) == "_no citations_"


def test_evidence_lines_list_the_gold_pages() -> None:
    assert evidence_lines([_citation(page=0), _citation(page=9)]) == "- `d/p000`\n- `d/p009`"


def test_evidence_lines_say_so_when_there_is_none() -> None:
    assert evidence_lines([]) == "_no gold evidence_"


def test_row_citations_parse_a_results_row() -> None:
    row = {"citations": [{"doc_id": "doc000", "page": 0, "bbox": None}]}

    (citation,) = row_citations(row)

    assert citation.doc_id == "doc000"
    assert citation.bbox is None


def test_row_citations_are_empty_without_the_key() -> None:
    """A REPL turn with no citations is a state the viewer must render, not crash on."""
    assert row_citations({}) == []


# ------------------------------------------------------------------ loading a results file


def _write_results(path: Path, run_mode: str, qid: str, answer: str) -> None:
    rows = [
        {"kind": "run", "mode": run_mode, "commit": "abc1234"},
        {
            "kind": "question",
            "qid": qid,
            "answer": answer,
            "citations": [
                {
                    "doc_id": "doc000",
                    "page": 0,
                    "bbox": {"x0": 0.1, "y0": 0.1, "x1": 0.2, "y1": 0.2},
                }
            ],
            "ndcg_at_k": 1.0,
        },
        {"kind": "summary", "n_done": 1},
    ]
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_runs_keeps_each_run_separate(tmp_path: Path) -> None:
    """Appended runs must not merge, or a viewer would show two arms as one."""
    path = tmp_path / "results.jsonl"
    _write_results(path, "vision", "q1", "vision answer")
    _write_results(path, "ocr", "q1", "ocr answer")

    runs = load_runs(path)

    assert [run["header"]["mode"] for run in runs] == ["vision", "ocr"]
    assert [run["questions"]["q1"]["answer"] for run in runs] == ["vision answer", "ocr answer"]


def test_load_runs_ignores_the_summary_as_a_question(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_results(path, "vision", "q1", "an answer")

    run = load_runs(path)[0]

    assert list(run["questions"]) == ["q1"]
    assert run["summary"]["n_done"] == 1


def test_append_turn_loads_without_gold_metrics(tmp_path: Path) -> None:
    """A REPL turn has citations and no gold, and that is enough to draw."""
    path = tmp_path / "turns.jsonl"
    append_turn(
        path,
        mode="vision",
        question="目标是什么",
        answer=Answer(text="四条", citations=[_citation(bbox=_box(0.1, 0.2, 0.3, 0.4))]),
        rounds=1,
        calls=4,
        tokens=10,
        stop_reason="",
    )

    row = load_runs(path)[0]["questions"]["目标是什么"]

    assert row["answer"] == "四条"
    assert "ndcg_at_k" not in row
    assert row["citations"][0]["bbox"]["x0"] == 0.1


def test_load_runs_on_a_missing_file_is_empty(tmp_path: Path) -> None:
    """An absent results file is a state the viewer must survive, not crash on."""
    assert load_runs(tmp_path / "nope.jsonl") == []


def test_load_runs_keeps_a_run_whose_summary_never_arrived(tmp_path: Path) -> None:
    """An aborted run is still worth viewing -- that is the point of flushing rows early."""
    path = tmp_path / "results.jsonl"
    with path.open("w") as handle:
        for row in (
            {"kind": "run", "mode": "vision"},
            {"kind": "question", "qid": "q1", "answer": "partial", "citations": []},
        ):
            handle.write(json.dumps(row) + "\n")

    runs = load_runs(path)

    assert runs[0]["questions"]["q1"]["answer"] == "partial"
    assert runs[0]["summary"] is None


def test_load_runs_skips_a_malformed_line(tmp_path: Path) -> None:
    """Rows are flushed as they are written, so a truncated line is a real state."""
    path = tmp_path / "results.jsonl"
    with path.open("w") as handle:
        handle.write(json.dumps({"kind": "run", "mode": "vision"}) + "\n")
        handle.write('{"kind": "question", "qid": "q1"\n')  # truncated JSON
        handle.write(json.dumps({"kind": "summary", "n_done": 0}) + "\n")

    runs = load_runs(path)

    assert runs[0]["summary"]["n_done"] == 0


def test_an_unknown_row_kind_does_not_create_a_phantom_run(tmp_path: Path) -> None:
    """A stray kind before the first run header must not swallow the questions after it."""
    path = tmp_path / "results.jsonl"
    with path.open("w") as handle:
        handle.write(json.dumps({"kind": "note"}) + "\n")
        handle.write(
            json.dumps({"kind": "question", "qid": "q1", "answer": "a", "citations": []}) + "\n"
        )

    runs = load_runs(path)

    assert runs[0]["questions"]["q1"]["answer"] == "a"


def test_page_image_returns_a_detached_copy(tmp_path: Path) -> None:
    """``Image.open`` is lazy; the caller must not be the one that releases the handle."""
    from multimodal_doc_qa.ui.viewer import page_image

    render = tmp_path / "render"
    (render / "d").mkdir(parents=True)
    Image.new("RGB", (10, 10), "white").save(render / "d" / "p000.png")

    image = page_image(render, "d", 0)
    assert image is not None
    (render / "d" / "p000.png").unlink()
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert page_image(render, "d", 1) is None


def test_runs_for_question_pairs_the_arms(tmp_path: Path) -> None:
    vision, ocr = tmp_path / "vision.jsonl", tmp_path / "ocr.jsonl"
    _write_results(vision, "vision", "q1", "vision answer")
    _write_results(ocr, "ocr", "q1", "ocr answer")

    paired = runs_for_question([*load_runs(vision), *load_runs(ocr)], "q1")

    assert [row["answer"] for _, row in paired] == ["vision answer", "ocr answer"]


def test_runs_for_question_skips_a_run_that_never_answered_it(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_results(path, "vision", "q1", "vision answer")

    paired = runs_for_question(load_runs(path), "q2")

    assert paired == []


# ------------------------------------------------------------------ the app itself


def test_the_app_renders_a_real_result_without_raising(tmp_path: Path) -> None:
    """The app is the deliverable, so "it runs" has to be checkable, not asserted.

    Streamlit executes the script on connect, so a served page proves nothing about the
    script. ``AppTest`` runs it for real and collects exceptions.
    """
    from multimodal_doc_qa.ui import viewer
    from streamlit.testing.v1 import AppTest

    artifacts = tmp_path / "artifacts"
    (artifacts / "render" / "doc000").mkdir(parents=True)
    Image.new("RGB", (200, 300), "white").save(artifacts / "render" / "doc000" / "p000.png")
    (artifacts / "questions.json").write_text(
        json.dumps(
            [
                {
                    "qid": "q1",
                    "text": "what is on page 0?",
                    "answer": "16.8%",
                    "evidence": [
                        {
                            "doc_id": "doc000",
                            "page": 0,
                            "bbox": {"x0": 0.1, "y0": 0.1, "x1": 0.3, "y1": 0.2},
                        }
                    ],
                    "hops": 1,
                }
            ]
        )
    )
    results = tmp_path / "results.jsonl"
    _write_results(results, "vision", "q1", "an answer")

    app = AppTest.from_file(viewer.__file__)
    app.run()
    app.text_input[0].set_value(str(artifacts))
    app.text_input[1].set_value(str(results))
    app.run()

    assert not app.exception
    assert app.selectbox[0].value == "q1"
    assert not app.warning


def test_the_app_draws_a_turn_that_has_no_gold(tmp_path: Path) -> None:
    """turns.jsonl from ask / the REPL has no questions.json and still paints the box."""
    from multimodal_doc_qa.ui import viewer
    from streamlit.testing.v1 import AppTest

    artifacts = tmp_path / "artifacts"
    (artifacts / "render" / "d").mkdir(parents=True)
    Image.new("RGB", (200, 300), "white").save(artifacts / "render" / "d" / "p000.png")
    append_turn(
        artifacts / "turns.jsonl",
        mode="vision",
        question="what is on the page?",
        answer=Answer(
            text="a box", citations=[_citation(doc_id="d", bbox=_box(0.1, 0.2, 0.3, 0.4))]
        ),
        rounds=1,
        calls=2,
        tokens=3,
        stop_reason="",
    )

    app = AppTest.from_file(viewer.__file__)
    app.run()
    app.text_input[0].set_value(str(artifacts))
    app.text_input[1].set_value("")
    app.text_input[2].set_value("")
    app.run()

    assert not app.exception
    assert app.selectbox[0].value == "what is on the page?"
    assert len(app.checkbox) == 0
    assert not app.warning


def test_the_app_says_what_to_do_before_it_has_inputs(tmp_path: Path) -> None:
    """No results file is the first state a reader meets; it must not look like a crash."""
    from multimodal_doc_qa.ui import viewer
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(viewer.__file__)
    app.run()
    app.text_input[0].set_value(str(tmp_path))
    app.text_input[1].set_value("")
    app.text_input[2].set_value("")
    app.run()

    assert not app.exception
    assert app.info[0].value.startswith("No turns yet")
