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
from multimodal_doc_qa.cli import IngestResult
from multimodal_doc_qa.config import Settings
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
    service = api.Service(
        settings=settings,
        mode="maxsim",
        artifacts=tmp_path,
        sessions=sessions or api.SessionStore(settings.session_ttl_seconds, settings.max_sessions),
    )
    # A pre-built arm and one catalog entry, so a turn proceeds without a checkpoint.
    service.retrievers["maxsim"] = object()
    service.documents = ["d"]
    return service


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
        "documents": 1,
    }


def test_an_empty_artifacts_directory_starts_without_a_retriever(tmp_path: Path) -> None:
    """An empty corpus is a valid state: the first upload creates the index.

    The server used to refuse to start here. It now boots in a second and loads the
    checkpoint on the upload that needs it, instead of holding 500M of weights for a
    corpus nobody has sent yet.
    """
    service = api.build_service(Settings(artifacts_dir=tmp_path))

    assert service.retrievers == {}
    assert service.documents == []
    assert service.encoder is None


def test_asking_an_empty_corpus_answers_without_running_the_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Saying "nothing ingested" must not spend model calls, and must not enter the history."""
    graph = _Graph(_echo)
    _wire(monkeypatch, graph)
    service = api.build_service(Settings(artifacts_dir=tmp_path))

    with TestClient(api.create_app(lambda: service)) as http:
        body = http.post("/ask", json={"question": "anything"}).json()

    assert body["stop_reason"] == "no_corpus"
    assert body["answer"]["text"] == api.EMPTY_CORPUS
    assert (body["calls"], body["rounds"]) == (0, 0)
    assert graph.states == []


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


# ------------------------------------------------------------------ /ingest


def _client_over(service: api.Service) -> TestClient:
    return TestClient(api.create_app(lambda: service))


def _upload(name: str, data: bytes = b"%PDF-1.4\n") -> tuple[str, tuple]:
    return "files", (name, data, "application/pdf")


def _stub_ingest(
    monkeypatch: pytest.MonkeyPatch, service: api.Service, result: IngestResult
) -> list[dict]:
    """Record the encoder/rebuild calls around the real one, which needs a checkpoint."""
    seen: list[dict] = []
    monkeypatch.setattr(service, "load_encoders", lambda: seen.append({"load": True}))

    def rebuild() -> None:
        service.documents += result.added
        seen.append({"rebuild": True})

    monkeypatch.setattr(service, "rebuild", rebuild)

    def fake(corpus, out=None, **kwargs):
        seen.append({"corpus": corpus, "out": out, **kwargs})
        return result

    monkeypatch.setattr(api, "run_ingest", fake)
    return seen


def test_an_upload_is_stored_and_encoded_with_the_loaded_encoders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A second ingest must not build a second copy of the checkpoint."""
    service = _service(tmp_path)
    service.documents = []  # an empty corpus: nothing indexed before this upload
    seen = _stub_ingest(monkeypatch, service, IngestResult(["paper"], [], 1, 3.5, 12.0))

    with _client_over(service) as http:
        body = http.post("/ingest", files=[_upload("paper.pdf")]).json()

    assert body == {
        "added": ["paper"],
        "skipped": [],
        "documents": 1,
        "index_mb": 3.5,
        "elapsed_s": 12.0,
    }
    assert (tmp_path / "corpus" / "paper.pdf").read_bytes() == b"%PDF-1.4\n"
    call = seen[1]
    assert call["append"] is True and call["out"] == tmp_path
    assert (call["vision_encoder"], call["ocr_embedder"]) == (service.encoder, service.embedder)
    assert seen[-1] == {"rebuild": True}

@pytest.mark.parametrize("name", ["tool.exe", "a", "notes.docx"])
def test_an_unsupported_upload_is_415(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    service = _service(tmp_path)
    _stub_ingest(monkeypatch, service, IngestResult([], [], 0, 0.0, 0.0))

    with _client_over(service) as http:
        assert http.post("/ingest", files=[_upload(name)]).status_code == 415


def test_an_upload_cannot_write_outside_the_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The client controls the name; a separator in it must not become a path."""
    service = _service(tmp_path)
    _stub_ingest(monkeypatch, service, IngestResult([], [], 0, 0.0, 0.0))

    with _client_over(service) as http:
        response = http.post("/ingest", files=[_upload("../../evil.pdf")])

    assert response.status_code == 200
    assert (tmp_path / "corpus" / "evil.pdf").is_file()
    assert not (tmp_path.parent / "evil.pdf").exists()


def test_an_empty_upload_is_400(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    service = _service(tmp_path)
    _stub_ingest(monkeypatch, service, IngestResult([], [], 0, 0.0, 0.0))

    with _client_over(service) as http:
        assert http.post("/ingest", files=[_upload("empty.pdf", b"")]).status_code == 400


# ------------------------------------------------------------------ delete


def _stub_remove(
    monkeypatch: pytest.MonkeyPatch, service: api.Service, remaining: list[str]
) -> list[dict]:
    seen: list[dict] = []
    monkeypatch.setattr(service, "load_encoders", lambda: seen.append({"load": True}))

    def rebuild() -> None:
        service.documents = remaining
        seen.append({"rebuild": True})

    monkeypatch.setattr(service, "rebuild", rebuild)

    def fake(doc_id, out=None, **kwargs):
        seen.append({"doc_id": doc_id, "out": out, **kwargs})
        return remaining

    monkeypatch.setattr(api, "remove_document", fake)
    return seen


def test_deleting_a_document_removes_it_and_lists_what_is_left(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same encoders as ingest, same lock, and a rebuild so the arms forget the old index."""
    service = _service(tmp_path)
    seen = _stub_remove(monkeypatch, service, ["e"])

    with _client_over(service) as http:
        body = http.request("DELETE", "/documents/paper").json()

    assert body == {"deleted": "paper", "documents": ["e"]}
    call = seen[1]
    assert call["doc_id"] == "paper" and call["out"] == tmp_path
    assert (call["vision_encoder"], call["ocr_embedder"]) == (service.encoder, service.embedder)
    assert seen[-1] == {"rebuild": True}


def test_deleting_an_unknown_document_is_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    service = _service(tmp_path)

    def fake(doc_id, out=None, **kwargs):
        raise KeyError(doc_id)

    monkeypatch.setattr(api, "remove_document", fake)

    with _client_over(service) as http:
        assert http.request("DELETE", "/documents/nope").status_code == 404


def test_a_document_ingested_from_elsewhere_is_400(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No source under artifacts/corpus means a rebuild would drop the other documents."""
    service = _service(tmp_path)

    def fake(doc_id, out=None, **kwargs):
        raise ValueError(f"{doc_id} has no source in {tmp_path / 'corpus'}")

    monkeypatch.setattr(api, "remove_document", fake)

    with _client_over(service) as http:
        response = http.request("DELETE", "/documents/paper")

    assert response.status_code == 400
    assert "no source" in response.json()["detail"]


# ------------------------------------------------------------------ static UI


def test_the_ui_is_served_at_the_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, _ = _client(monkeypatch, tmp_path, {})

    with client as http:
        response = http.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "/ask" in response.text and "/ingest" in response.text and "/health" in response.text


def test_documents_lists_what_the_current_index_holds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    service = _service(tmp_path)
    service.documents = ["d", "e"]

    with _client_over(service) as http:
        body = http.get("/documents").json()

    assert body == {"mode": "maxsim", "documents": ["d", "e"]}


# ------------------------------------------------------------------ retrieval arms


def _stub_build(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record each arm build and return a distinguishable retriever."""
    built: list[str] = []

    def fake(settings, mode, **kwargs):
        built.append(mode)
        return {"arm": mode}

    monkeypatch.setattr("multimodal_doc_qa.retrievers.assembly.build_retriever", fake)
    return built


def test_the_ask_uses_the_arm_it_was_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Switching the arm switches the retriever the graph retrieves with."""
    built = _stub_build(monkeypatch)
    service = _service(tmp_path)
    service.retrievers.clear()
    graph = _Graph(_echo)
    handed: list[object] = []
    monkeypatch.setattr(
        api, "load_deps", lambda settings, retriever, budget, **kw: handed.append(retriever) or object()
    )
    monkeypatch.setattr(api, "build_graph", lambda deps, settings: graph)

    with _client_over(service) as http:
        body = http.post("/ask", json={"question": "hi", "mode": "lexical"}).json()

    assert body["mode"] == "lexical"
    assert built == ["lexical"]
    assert handed == [{"arm": "lexical"}]
    assert graph.states[0]["messages"][-1].content == "hi"


def test_one_arm_is_built_once_and_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two turns on the same arm must not pay for a second index load."""
    built = _stub_build(monkeypatch)
    service = _service(tmp_path)
    service.retrievers.clear()
    _wire(monkeypatch, _Graph(_echo))

    with _client_over(service) as http:
        http.post("/ask", json={"question": "one", "mode": "ocr"})
        http.post("/ask", json={"question": "two", "mode": "ocr"})

    assert built == ["ocr"]


def test_a_rebuild_drops_the_arms_built_from_the_old_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built = _stub_build(monkeypatch)
    (tmp_path / "documents.json").write_text(
        '[{"doc_id": "d", "origin": "pdf"}]', encoding="utf-8"
    )
    service = _service(tmp_path)
    service.retrievers.clear()
    service.retriever_for("ocr")

    service.rebuild()

    # The default arm is built eagerly for the new index; the stale 'ocr' one is gone.
    assert built == ["ocr", "maxsim"]
    assert list(service.retrievers) == ["maxsim"]


def test_modes_lists_only_the_arms_the_index_can_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The abstract arms need a VLM pass at ingest; a plain corpus must not offer them."""
    service = _service(tmp_path)
    for name in ("multivector_index.pt", "ocr_index.pt"):
        (tmp_path / name).write_bytes(b"")

    with _client_over(service) as http:
        body = http.get("/modes").json()

    assert body["default"] == "maxsim"
    assert "maxsim" in body["modes"] and "hybrid-maxsim" in body["modes"]
    assert "abstract" not in body["modes"] and "hybrid-abstract" not in body["modes"]


def test_no_index_means_no_arms(tmp_path: Path) -> None:
    assert api.available_modes(Settings(artifacts_dir=tmp_path)) == []


def test_an_unknown_mode_is_422(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_build(monkeypatch)
    client, _ = _client(monkeypatch, tmp_path, {})

    with client as http:
        response = http.post("/ask", json={"question": "hi", "mode": "nope"})

    assert response.status_code == 422
    assert "nope" in response.json()["detail"]


def test_an_arm_whose_index_is_missing_is_400(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real arm this corpus never built (abstract) must say so, not 500 mid-turn."""
    client, _ = _client(monkeypatch, tmp_path, {})

    with client as http:
        response = http.post("/ask", json={"question": "hi", "mode": "abstract"})

    assert response.status_code == 400
    assert "abstract" in response.json()["detail"]
