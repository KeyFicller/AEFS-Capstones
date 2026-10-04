"""These nodes decide whether the agentic loop keeps going or stops, so the tests pin the
two properties that are easy to lose while every plumbing test still passes.

First, output is structured rather than hand-parsed: the stubs hand back the schema
instance a bound model would, so a node that tried to ``json.loads`` it would fail here.

Second, ``assess`` and ``verify`` are shown the pages. Given only page ids they cannot do
their jobs at all -- and nothing else in the suite would notice.
"""

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from multimodal_doc_qa.agent.nodes import Followups, Subqueries, Unsupported, assess, plan, verify
from multimodal_doc_qa.schemas import Answer, Citation
from PIL import Image


class _StubModel:
    """Hands back the schema instance a schema-bound model would, recording what it saw."""

    def __init__(self, result: object = None) -> None:
        self.result = result
        self.calls: list[list[dict]] = []

    def invoke(self, messages: list[dict]) -> object:
        self.calls.append(messages)
        return self.result


def _page(tmp_path: Path, page: str, size: tuple[int, int] = (4, 3)) -> None:
    path = tmp_path / f"{page}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def _ask(text: str) -> list[HumanMessage]:
    return [HumanMessage(content=text)]


def _content(message: object) -> object:
    if isinstance(message, dict):
        return message["content"]
    return message.content  # type: ignore[attr-defined]


def _json_text(messages: list) -> str:
    """The text part of the user turn, i.e. everything a node can say without pixels."""
    content = _content(messages[-1])
    if isinstance(content, str):
        return content
    return "\n".join(p["text"] for p in content if p["type"] == "text")


def _images(messages: list) -> list[dict]:
    content = _content(messages[-1])
    if isinstance(content, str):
        return []
    return [p for p in content if p["type"] == "image_url"]


def _urls(messages: list[dict]) -> list[str]:
    return [p["image_url"]["url"] for p in _images(messages)]


# ---------------------------------------------------------------------------- plan


def test_plan_returns_the_subqueries_the_model_produced() -> None:
    model = _StubModel(Subqueries(subqueries=["EMEA margin", "APAC margin"]))

    assert plan(model, _ask("compare margins")) == ["EMEA margin", "APAC margin"]


def test_plan_falls_back_to_the_question_when_nothing_is_decomposed() -> None:
    """An empty plan would leave ``retrieve`` with nothing to search, stranding the ask."""
    model = _StubModel(Subqueries(subqueries=[]))

    assert plan(model, _ask("compare margins")) == ["compare margins"]


def test_plan_asks_without_any_page() -> None:
    model = _StubModel(Subqueries(subqueries=["x"]))

    plan(model, _ask("compare margins"))

    assert "compare margins" in _json_text(model.calls[0])
    assert _images(model.calls[0]) == []


# -------------------------------------------------------------------------- assess


def test_assess_returns_the_followups() -> None:
    model = _StubModel(Followups(followups=["which regions?"]))

    assert assess(model, _ask("margin?"), [], Path()) == ["which regions?"]


def test_assess_empty_means_the_pool_is_sufficient() -> None:
    model = _StubModel(Followups(followups=[]))

    assert assess(model, _ask("margin?"), [], Path()) == []


def test_assess_shows_the_page_images_not_just_their_ids(tmp_path: Path) -> None:
    """Sufficiency cannot be judged from ids: they say nothing about what is on a page."""
    _page(tmp_path, "doc000/p000")
    _page(tmp_path, "doc000/p001")
    model = _StubModel(Followups(followups=[]))

    assess(model, _ask("margin?"), ["doc000/p000", "doc000/p001"], tmp_path)

    assert len(_images(model.calls[0])) == 2
    assert "doc000/p000" in _json_text(model.calls[0])


# -------------------------------------------------------------------------- verify


def _answer(*citations: Citation) -> Answer:
    return Answer(text="The EMEA margin was 16.8%.", citations=list(citations))


def test_verify_returns_the_claims_the_model_called_unsupported(tmp_path: Path) -> None:
    _page(tmp_path, "doc000/p000")
    model = _StubModel(Unsupported(unsupported=["it covers the enterprise segment"]))
    answer = _answer(Citation(doc_id="doc000", page=0))

    result = verify(model, _ask("margin?"), answer, ["doc000/p000"], tmp_path)

    assert result == ["it covers the enterprise segment"]


def test_verify_shows_the_cited_page_images(tmp_path: Path) -> None:
    """A claim can only be checked against a page the model can actually read."""
    _page(tmp_path, "doc000/p002")
    model = _StubModel(Unsupported(unsupported=[]))

    verify(
        model,
        _ask("margin?"),
        _answer(Citation(doc_id="doc000", page=2)),
        ["doc000/p002"],
        tmp_path,
    )

    assert len(_images(model.calls[0])) == 1


def test_verify_passes_when_every_citation_is_supported(tmp_path: Path) -> None:
    _page(tmp_path, "doc000/p000")
    model = _StubModel(Unsupported(unsupported=[]))

    assert (
        verify(
            model,
            _ask("margin?"),
            _answer(Citation(doc_id="doc000", page=0)),
            ["doc000/p000"],
            tmp_path,
        )
        == []
    )


def test_verify_flags_a_citation_outside_the_pool_without_asking_the_model(
    tmp_path: Path,
) -> None:
    """The answerer invents ``doc_id`` values, so membership must not rely on the model.

    Version of a claim from a page the retriever never returned is unsupported by
    construction, and the check has to hold even when the model reports no problem.
    """
    model = _StubModel(Unsupported(unsupported=[]))
    answer = _answer(Citation(doc_id="doc000", page=999))

    result = verify(model, _ask("margin?"), answer, ["doc000/p000"], tmp_path)

    assert result
    assert model.calls == []


def test_verify_does_not_ask_the_model_when_no_citation_is_inside_the_pool(
    tmp_path: Path,
) -> None:
    """There is no page to show, so there is nothing to ask about."""
    model = _StubModel(Unsupported(unsupported=["ignored"]))
    answer = _answer(Citation(doc_id="doc000", page=999))

    result = verify(model, _ask("margin?"), answer, ["doc000/p000"], tmp_path)

    assert result
    assert model.calls == []


def test_verify_reports_an_answer_that_cites_nothing(tmp_path: Path) -> None:
    """An unverifiable answer is not a supported one; silence here would pass it as verified."""
    model = _StubModel(Unsupported(unsupported=[]))

    result = verify(model, _ask("margin?"), _answer(), ["doc000/p000"], tmp_path)

    assert result == ["answer cites no page"]
    assert model.calls == []


def test_plan_passes_prior_turns_as_messages() -> None:
    """History stays a message list. Nothing joins it into the question string."""
    model = _StubModel(Subqueries(subqueries=["x"]))

    plan(
        model,
        [
            HumanMessage(content="earlier question"),
            AIMessage(content="earlier answer"),
            HumanMessage(content="follow up"),
        ],
    )

    sent = model.calls[0]
    assert [message.type for message in sent] == ["system", "human", "ai", "human"]
    assert sent[-1].content == "follow up"
    assert all("Earlier turns" not in str(message.content) for message in sent)
