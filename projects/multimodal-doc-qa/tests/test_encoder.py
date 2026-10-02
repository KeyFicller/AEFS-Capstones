import pytest
from PIL import Image

from multimodal_doc_qa.config import Settings
from multimodal_doc_qa.embed import encoder as encoder_module
from multimodal_doc_qa.embed.encoder import MultiVectorEncoder


def test_settings_defaults() -> None:
    """The defaults are the project's declared setup; later tasks read them from here.

    ``conftest`` clears ambient ``MDQ_*`` overrides, so this reads the declared defaults
    rather than whatever the developer happens to have exported.
    """
    settings = Settings()
    assert settings.embedder_model == "vidore/colqwen2.5-v0.2"
    assert settings.embedder_fallback == "vidore/colSmol-500M"
    assert settings.device == "mps"
    assert settings.dtype == "float16"
    assert settings.top_k == 5
    assert settings.max_rounds == 3
    assert settings.max_ask_calls == 10
    assert settings.max_ask_tokens == 80_000
    assert settings.max_ask_seconds == 120.0


def test_settings_reads_mdq_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No tunable may be hardcoded: the budget and device must be overridable per run."""
    monkeypatch.setenv("MDQ_TOP_K", "9")
    monkeypatch.setenv("MDQ_DEVICE", "cpu")
    settings = Settings()
    assert settings.top_k == 9
    assert settings.device == "cpu"


def test_encoder_falls_back_when_primary_load_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """MPS OOM and download failures are expected, so the fallback branch must work."""
    calls: list[str] = []

    def fake_load(model_name: str, settings: Settings) -> tuple[str, str]:
        calls.append(model_name)
        if len(calls) == 1:
            raise OSError("simulated checkpoint download failure")
        return "model", "processor"

    monkeypatch.setattr(encoder_module, "_load", fake_load)
    settings = Settings(embedder_model="vidore/colqwen2.5-v0.2")

    encoder = MultiVectorEncoder(settings.embedder_model, settings)

    assert calls == [settings.embedder_model, settings.embedder_fallback]
    assert encoder.model == "model"


@pytest.mark.slow
def test_encoder_shapes_and_device() -> None:
    settings = Settings(embedder_model="vidore/colSmol-500M", device="mps", dtype="float16")
    enc = MultiVectorEncoder(settings.embedder_model, settings)
    pages = enc.encode_images([Image.new("RGB", (512, 384), "white")])
    assert len(pages) == 1 and pages[0].ndim == 2 and pages[0].shape[1] == 128
    q = enc.encode_query("what was the margin?")
    assert q.ndim == 2 and q.shape[1] == 128
