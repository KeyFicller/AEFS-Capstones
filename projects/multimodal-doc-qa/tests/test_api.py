"""The HTTP layer: routing, response shape, page overlay, and the conversation store.

No checkpoint is loaded: the graph is a stub, so these tests measure this module only.
"""

import io
import threading
import time
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage
from multimodal_doc_qa import api
from multimodal_doc_qa.config import Settings
from multimodal_doc_qa.retrievers.assembly import IndexNotFoundError
from multimodal_doc_qa.schemas import Answer, Citation
from multimodal_doc_qa.ui.viewer import CITED
from PIL import Image


class _Graph:
    """Stands in for the compiled loop: records the state it was handed."""

    def __init__(self, reply: dict | object, delay: float = 0.0) -> None:
        self.reply = reply
        self.delay = delay
        self.states: list[dict] = []

    def invoke(self, state: dict) -> dict:
        self.states.append(state)
        if self.delay:
            time.sleep(self.delay)
        return self.reply(state) if callable(self.reply) else self.reply


def _echo(state: dict) -> dict:
    """Answer whatever was asked, so a test can see which turn produced which reply."""
    question = state["messages"][-1].content
    return {
        "answer": {"text": f"re: {question}", "citations": []},
        "rounds": 1,
        "stop_reason": "",
    }


def _service(tmp_path: Path, sessions: api.SessionStore | None = None) -> api.Service:
    settings = Settings(artifacts_dir=tmp_path)
    return api.Service(
        settings=settings,
        retriever=object(),
        mode="maxsim",
        artifacts=tmp_path,
        sessions=sessions or api.SessionStore(settings.session_ttl_seconds, settings.max_sessions),
    )


def _wire(monkeypatch: pytest.MonkeyPatch, graph: _Graph) -> None:
    monkeypatch.setattr(api, "load_deps", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "build_graph", lambda deps, settings: graph)


def _client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reply: dict | object
) -> tuple[TestClient, _Graph]:
    """A client over a fake graph, plus the graph so a test can inspect what it received."""
    service = _service(tmp_path)
    graph = _Graph(reply)
    _wire(monkeypatch, graph)
    return TestClient(api.create_app(lambda: service)), graph


# ------------------------------------------------------------------ /ask


def test_ask_returns_the_answer_its_citations_and_what_it_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer = Answer(text="16.8%", citations=[Citation(doc_id="d", page=2)])
    client, graph = _client(
        monkeypatch, tmp_path, {"answer": answer.model_dump(), "rounds": 2, "stop_reason": ""}
    )

    with client as http:
        response = http.post("/ask", json={"question": "what share?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == answer.model_dump()
    assert (body["rounds"], body["mode"], body["stop_reason"]) == (2, "maxsim", "")
    assert body["calls"] == 0
    assert body["latency_s"] >= 0
    assert graph.states[0]["messages"][-1].content == "what share?"


def test_an_empty_pool_is_a_200_with_its_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing retrieved is an answer about the corpus, not a transport failure."""
    client, _ = _client(
        monkeypatch, tmp_path, {"answer": None, "rounds": 1, "stop_reason": "recover_empty"}
    )

    with client as http:
        response = http.post("/ask", json={"question": "nothing here"})

    assert response.status_code == 200
    assert response.json()["answer"] is None
    assert response.json()["stop_reason"] == "recover_empty"


def test_an_empty_question_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, _ = _client(monkeypatch, tmp_path, {"answer": None, "rounds": 0, "stop_reason": ""})

    with client as http:
        assert http.post("/ask", json={"question": ""}).status_code == 422


def test_health_names_the_arm_the_artifacts_and_the_session_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client, _ = _client(monkeypatch, tmp_path, {})

    with client as http:
        body = http.get("/health").json()

    assert body == {
        "status": "ok",
        "mode": "maxsim",
        "artifacts": str(tmp_path),
        "sessions": 0,
    }


def test_a_missing_index_fails_at_startup(tmp_path: Path) -> None:
    """Fail where a log line can explain it, not on the first question."""
    with pytest.raises(IndexNotFoundError):
        api.build_service(Settings(artifacts_dir=tmp_path))


# ------------------------------------------------------------------ conversations


def test_a_second_turn_sees_the_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole point: the follow-up must arrive with the earlier question and answer."""
    graph = _Graph(_echo)
    _wire(monkeypatch, graph)
    service = _service(tmp_path)

    first = api.answer_question(service, "what share was that?")
    second = api.answer_question(service, "and the year before?", first.session_id)

    assert second.session_id == first.session_id
    seen = [message.content for message in graph.states[1]["messages"]]
    assert seen == ["what share was that?", "re: what share was that?", "and the year before?"]


def test_conversations_do_not_leak_into_each_other(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph = _Graph(_echo)
    _wire(monkeypatch, graph)
    service = _service(tmp_path)

    api.answer_question(service, "one")
    other = api.answer_question(service, "unrelated")
    api.answer_question(service, "follow up", other.session_id)

    assert [message.content for message in graph.states[-1]["messages"]] == [
        "unrelated",
        "re: unrelated",
        "follow up",
    ]


def test_omitting_the_id_starts_one_and_returns_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph = _Graph(_echo)
    _wire(monkeypatch, graph)
    service = _service(tmp_path)

    first = api.answer_question(service, "hello")
    graph.states.clear()
    api.answer_question(service, "again", first.session_id)

    assert first.session_id
    assert len(graph.states[0]["messages"]) == 3


def test_an_unknown_id_starts_a_fresh_conversation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A restart wipes the store; clients must not have to handle a 404 for that."""
    graph = _Graph(_echo)
    _wire(monkeypatch, graph)
    service = _service(tmp_path)

    response = api.answer_question(service, "hello", "never-seen")

    assert response.session_id == "never-seen"
    assert len(graph.states[0]["messages"]) == 1


def test_a_turn_without_an_answer_is_not_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A question with no reply must not sit in front of the next turn."""
    graph = _Graph({"answer": None, "rounds": 1, "stop_reason": "recover_empty"})
    _wire(monkeypatch, graph)
    service = _service(tmp_path)

    first = api.answer_question(service, "nothing here")
    graph.states.clear()
    api.answer_question(service, "next", first.session_id)

    assert len(graph.states[0]["messages"]) == 1


def test_the_store_expires_an_idle_session() -> None:
    now = [0.0]
    store = api.SessionStore(ttl=60, capacity=10, clock=lambda: now[0])
    _, session = store.get_or_create("a")
    session.history.append(HumanMessage(content="old"))

    now[0] = 61
    _, fresh = store.get_or_create("a")

    assert fresh is not session
    assert fresh.history == []


def test_the_store_evicts_the_least_recently_used_over_capacity() -> None:
    now = [0.0]
    store = api.SessionStore(ttl=10_000, capacity=2, clock=lambda: now[0])
    store.get_or_create("a")
    now[0] = 1
    store.get_or_create("b")
    now[0] = 2
    store.get_or_create("c")

    assert len(store) == 2
    _, session = store.get_or_create("a")
    assert session.history == []  # "a" was evicted, so this is a new session


def test_two_turns_of_one_conversation_are_serialized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without a per-session lock both turns read the same empty history and one is lost."""
    graph = _Graph(_echo, delay=0.05)
    _wire(monkeypatch, graph)
    service = _service(tmp_path)

    threads = [
        threading.Thread(target=api.answer_question, args=(service, question, "same"))
        for question in ("first", "second")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(service.sessions) == 1
    assert [len(state["messages"]) for state in graph.states] == [1, 3]


def test_different_conversations_run_in_parallel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A global lock would make this take two turns' worth of wall clock."""
    graph = _Graph(_echo, delay=0.3)
    _wire(monkeypatch, graph)
    service = _service(tmp_path)

    started = time.monotonic()
    threads = [
        threading.Thread(target=api.answer_question, args=(service, question))
        for question in ("alpha", "beta")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert time.monotonic() - started < 0.5


# ------------------------------------------------------------------ /page


def _render(tmp_path: Path) -> None:
    (tmp_path / "render" / "d").mkdir(parents=True)
    Image.new("RGB", (100, 200), "white").save(tmp_path / "render" / "d" / "p000.png")


def test_page_draws_the_requested_box(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The page is twice as tall as it is wide, so an x/y swap lands on empty pixels."""
    _render(tmp_path)
    client, _ = _client(monkeypatch, tmp_path, {})

    with client as http:
        response = http.get("/page/d/0.png", params={"bbox": "0.1,0.2,0.3,0.4"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    image = Image.open(io.BytesIO(response.content))
    assert image.getpixel((10, 40)) == CITED
    assert image.getpixel((40, 10)) == (255, 255, 255)


def test_several_boxes_land_on_one_page(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _render(tmp_path)
    client, _ = _client(monkeypatch, tmp_path, {})

    with client as http:
        response = http.get(
            "/page/d/0.png",
            params=[("bbox", "0.1,0.1,0.2,0.2"), ("bbox", "0.5,0.5,0.6,0.6")],
        )

    image = Image.open(io.BytesIO(response.content))
    assert image.getpixel((10, 20)) == CITED
    assert image.getpixel((50, 100)) == CITED


def test_page_without_boxes_is_not_marked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No box means nothing was located; the page must not look like a cited region."""
    _render(tmp_path)
    client, _ = _client(monkeypatch, tmp_path, {})

    with client as http:
        response = http.get("/page/d/0.png")

    image = Image.open(io.BytesIO(response.content))
    assert image.getpixel((10, 40)) == (255, 255, 255)


def test_an_unrendered_page_is_404(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _render(tmp_path)
    client, _ = _client(monkeypatch, tmp_path, {})

    with client as http:
        assert http.get("/page/d/7.png").status_code == 404


@pytest.mark.parametrize("raw", ["0.1,0.2", "2,0,1,1", "a,b,c,d"])
def test_a_bad_box_is_422(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: str) -> None:
    _render(tmp_path)
    client, _ = _client(monkeypatch, tmp_path, {})

    with client as http:
        assert http.get("/page/d/0.png", params={"bbox": raw}).status_code == 422


@pytest.mark.parametrize("doc_id", ["..", ".", "a/b", "a\\b"])
def test_a_doc_id_cannot_escape_the_render_directory(tmp_path: Path, doc_id: str) -> None:
    """The id becomes a path segment; a separator would read outside the artifacts."""
    with pytest.raises(HTTPException) as raised:
        api._page_png(_service(tmp_path), doc_id, 0, None)

    assert raised.value.status_code == 404
