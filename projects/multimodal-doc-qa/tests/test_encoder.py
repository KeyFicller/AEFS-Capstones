import pytest
import torch
from multimodal_doc_qa.config import Settings
from multimodal_doc_qa.embed import encoder as encoder_module
from multimodal_doc_qa.embed.encoder import MultiVectorEncoder
from PIL import Image


def test_settings_defaults() -> None:
    """The defaults are the project's declared setup; later tasks read them from here.

    ``conftest`` clears ambient ``MDQ_*`` overrides, so this reads the declared defaults
    rather than whatever the developer happens to have exported.
    """
    settings = Settings()
    assert settings.embedder_model == "vidore/colSmol-500M"
    assert settings.ocr_embedder_model == "BAAI/bge-small-en-v1.5"
    assert settings.device == "mps"
    assert settings.dtype == "float16"
    assert settings.top_k == 5
    assert settings.min_score_ratio == 0.5
    assert settings.max_rounds == 5
    assert settings.max_ask_calls == 16
    assert settings.max_ask_tokens == 200_000
    assert settings.max_ask_seconds == 120.0


def test_settings_reads_mdq_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No tunable may be hardcoded: the budget and device must be overridable per run."""
    monkeypatch.setenv("MDQ_TOP_K", "9")
    monkeypatch.setenv("MDQ_DEVICE", "cpu")
    settings = Settings()
    assert settings.top_k == 9
    assert settings.device == "cpu"


def test_encode_texts_calls_process_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document text must not pick up the query-augmentation suffix."""

    class _Batch(dict):
        def to(self, device: str) -> _Batch:
            return self

    class _Processor:
        def process_texts(self, texts: list[str]) -> _Batch:
            assert texts == ["The tanh gate starts at zero."]
            return _Batch()

        def process_queries(self, texts: list[str]) -> _Batch:
            raise AssertionError("document text went through the query prompt")

    class _Model:
        def __call__(self, **batch: object) -> torch.Tensor:
            return torch.zeros(1, 4, 128)

    monkeypatch.setattr(encoder_module, "_load", lambda *a, **k: (_Model(), _Processor()))
    encoder = MultiVectorEncoder("vidore/colSmol-500M", Settings(device="cpu"))

    encoded = encoder.encode_texts(["The tanh gate starts at zero."])

    assert encoded[0].shape == (4, 128)
    assert encoded[0].dtype == torch.float32


def test_encoder_load_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed checkpoint load is an error. There is no second model to switch to."""

    def fake_load(model_name: str, settings: Settings) -> tuple[str, str]:
        raise OSError("simulated checkpoint download failure")

    monkeypatch.setattr(encoder_module, "_load", fake_load)

    with pytest.raises(OSError, match="simulated checkpoint download failure"):
        MultiVectorEncoder(Settings().embedder_model, Settings())


def test_an_unknown_dtype_names_the_accepted_values() -> None:
    """The dtype is a free-form config string; a typo must not surface as a bare KeyError."""
    assert encoder_module._resolve_dtype("float32") is torch.float32
    with pytest.raises(ValueError, match="unknown dtype"):
        encoder_module._resolve_dtype("fp16")


@pytest.mark.slow
def test_encoder_shapes_and_device() -> None:
    settings = Settings(embedder_model="vidore/colSmol-500M", device="mps", dtype="float16")
    enc = MultiVectorEncoder(settings.embedder_model, settings)
    pages = enc.encode_images([Image.new("RGB", (512, 384), "white")])
    assert len(pages) == 1 and pages[0].ndim == 2 and pages[0].shape[1] == 128
    q = enc.encode_query("what was the margin?")
    assert q.ndim == 2 and q.shape[1] == 128
