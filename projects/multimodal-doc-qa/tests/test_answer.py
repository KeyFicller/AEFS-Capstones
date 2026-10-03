"""Citations are a hard contract, so these tests pin both halves of it.

The answerer is bound to ``Answer`` through structured output, and a reply that cannot
satisfy the schema fails the ask -- there is deliberately no repair path. What stays
optional is localization: a citation may omit ``bbox`` and be page-level, so the model
can always decline to guess a box. The tests below hold that line: a missing box is a
valid answer, a malformed one is not.
"""

import base64
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from multimodal_doc_qa.config import Settings
from multimodal_doc_qa.schemas import Answer, Citation, TextDocument
from multimodal_doc_qa.synth.answer import AnswerSynthesizer, build_page_blocks
from PIL import Image
from pydantic import ValidationError


def _page(tmp_path: Path, page_id: str, size: tuple[int, int] = (4, 3)) -> Path:
    path = tmp_path / f"{page_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


def _inverted_bbox_error() -> ValidationError:
    """The schema's own rejection of a box the model cannot have meant to send."""
    with pytest.raises(ValidationError) as info:
        Answer.model_validate(
            {
                "text": "16.8%",
                "citations": [
                    {
                        "doc_id": "doc000",
                        "page": 2,
                        "bbox": {"x0": 0.5, "y0": 0.5, "x1": 0.1, "y1": 0.1},
                    }
                ],
            }
        )
    return info.value


# ------------------------------------------------------------- build_page_blocks


def test_build_page_blocks_labels_each_page_before_its_image(tmp_path: Path) -> None:
    _page(tmp_path, "doc000/p000")
    _page(tmp_path, "doc000/p001")

    blocks = build_page_blocks(["doc000/p000", "doc000/p001"], tmp_path)

    assert [b["type"] for b in blocks] == ["text", "image_url", "text", "image_url"]
    assert [b["text"] for b in blocks if b["type"] == "text"] == [
        "page doc000/p000",
        "page doc000/p001",
    ]


def test_build_page_blocks_inlines_the_png_bytes(tmp_path: Path) -> None:
    png = _page(tmp_path, "doc000/p000")

    blocks = build_page_blocks(["doc000/p000"], tmp_path)

    expected = "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()
    assert blocks[1]["image_url"]["url"] == expected


def test_build_page_blocks_does_not_send_a_file_path(tmp_path: Path) -> None:
    """A path would be meaningless to a hosted model; the pixels have to travel."""
    png = _page(tmp_path, "doc000/p000")

    url = build_page_blocks(["doc000/p000"], tmp_path)[1]["image_url"]["url"]

    assert url.startswith("data:image/png;base64,")
    assert str(png) not in url


def test_build_page_blocks_reports_a_missing_render(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_page_blocks(["doc000/p999"], tmp_path)


def test_build_page_blocks_sends_text_for_a_text_document(tmp_path: Path) -> None:
    blocks = build_page_blocks(
        ["note/p000"],
        tmp_path,
        {"note": TextDocument(doc_id="note", pages=["The tanh gate starts at zero."])},
    )

    assert [b["type"] for b in blocks] == ["text", "text"]
    assert blocks[1]["text"] == "The tanh gate starts at zero."


# ------------------------------------------------------------ AnswerSynthesizer


class _UnsupportedStructuredOutput(RuntimeError):
    """Stands in for a provider that cannot bind a schema.

    A distinct type on purpose: asserting ``NotImplementedError`` here would also
    match an unimplemented scaffold, so the test would pass without the feature
    existing at all.
    """


class _StubStructured:
    """Stands in for the provider's structured-output runnable."""

    def __init__(self, result: object = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[list[dict]] = []

    def invoke(self, messages: list[dict]) -> object:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.result


class _StubModel:
    """Stands in for the chat model.

    ``invoke`` fails loudly: with structured output required, reaching the unparsed
    call is itself the bug, so a test that triggers it should die on the spot.
    """

    def __init__(
        self,
        structured: _StubStructured | None = None,
        unsupported: Exception | None = None,
    ) -> None:
        self._structured = structured
        self._unsupported = unsupported
        self.plain_calls: list[list[dict]] = []

    def with_structured_output(self, schema: object) -> _StubStructured:
        if self._unsupported is not None:
            raise self._unsupported
        return self._structured or _StubStructured()

    def invoke(self, messages: list[dict]) -> object:
        self.plain_calls.append(messages)
        raise AssertionError("the synthesizer must not fall back to an unparsed call")


def _synthesizer(monkeypatch: pytest.MonkeyPatch, model: _StubModel) -> AnswerSynthesizer:
    """Build a synthesizer over ``model`` without touching a provider or an API key."""
    monkeypatch.setattr("langchain.chat_models.init_chat_model", lambda *a, **k: model)
    return AnswerSynthesizer(Settings())


def test_synthesize_returns_the_structured_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wanted = Answer(
        text="16.8%",
        citations=[Citation(doc_id="doc000", page=2, bbox=None)],
    )
    model = _StubModel(structured=_StubStructured(result=wanted))
    _page(tmp_path, "doc000/p002")

    answer = _synthesizer(monkeypatch, model).synthesize(
        [HumanMessage(content="margin?")], ["doc000/p002"], tmp_path
    )

    assert answer == wanted
    assert model.plain_calls == []


def test_synthesize_accepts_a_page_level_citation_without_a_bbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declining to localize is a valid answer: the schema makes bbox optional."""
    page_level = Answer(text="x", citations=[Citation(doc_id="doc000", page=1, bbox=None)])
    model = _StubModel(structured=_StubStructured(result=page_level))
    _page(tmp_path, "doc000/p001")

    answer = _synthesizer(monkeypatch, model).synthesize(
        [HumanMessage(content="margin?")], ["doc000/p001"], tmp_path
    )

    assert answer.citations[0].bbox is None


def test_synthesize_sends_the_question_then_each_page_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structured = _StubStructured(result=Answer(text="x"))
    model = _StubModel(structured=structured)
    _page(tmp_path, "doc000/p000")
    _page(tmp_path, "doc000/p001")

    _synthesizer(monkeypatch, model).synthesize(
        [HumanMessage(content="which margin?")], ["doc000/p000", "doc000/p001"], tmp_path
    )

    sent = structured.calls[0]
    assert [m.type for m in sent] == ["system", "human"]
    parts = sent[1].content
    assert parts[0] == {"type": "text", "text": "which margin?"}
    assert [p["type"] for p in parts] == ["text", "text", "image_url", "text", "image_url"]
    assert [p["text"] for p in parts if p["type"] == "text"] == [
        "which margin?",
        "page doc000/p000",
        "page doc000/p001",
    ]


def test_synthesize_propagates_an_invalid_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed citation fails the ask; it must not be repaired or retried."""
    model = _StubModel(structured=_StubStructured(error=_inverted_bbox_error()))
    _page(tmp_path, "doc000/p002")

    with pytest.raises(ValidationError):
        _synthesizer(monkeypatch, model).synthesize(
            [HumanMessage(content="margin?")], ["doc000/p002"], tmp_path
        )

    assert model.plain_calls == []


def test_synthesizer_surfaces_an_unsupported_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structured output is required, so an unbound model must fail at construction."""
    model = _StubModel(unsupported=_UnsupportedStructuredOutput("no tool calling"))

    with pytest.raises(_UnsupportedStructuredOutput):
        _synthesizer(monkeypatch, model)


def test_synthesizer_disables_thinking_mode_on_the_chat_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thinking mode rejects every forced ``tool_choice``, so the bound schema 400s.

    Pinned here because the failure is invisible in this file: the flag reads like an
    optional provider tweak, and dropping it turns every real ask into an API error
    while all the stubbed tests keep passing.
    """
    captured: dict = {}
    model = _StubModel()

    def fake_init(*args: object, **kwargs: object) -> _StubModel:
        captured["args"] = args
        captured.update(kwargs)
        return model

    monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init)
    AnswerSynthesizer(Settings())

    assert captured["args"] == ("deepseek:deepseek-flash",)
    assert "model_provider" not in captured
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
