"""Runtime configuration for the whole pipeline.

Every tunable lives here rather than as a literal in business logic, so the
device, the models and the budget caps can be overridden per run.
"""

from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def artifact_paths(root: Path) -> tuple[Path, Path, Path]:
    """``(render_dir, vision_index, ocr_index)`` under ``root``. One layout for ingest and ask."""
    return root / "render", root / "vision_index.pt", root / "ocr_index.pt"


class Settings(BaseSettings):
    """Settings from the environment, prefix ``MDQ_``."""

    model_config = SettingsConfigDict(env_prefix="MDQ_", extra="ignore")

    embedder_model: str = "vidore/colqwen2.5-v0.2"
    embedder_fallback: str = "vidore/colSmol-500M"
    answerer_provider: str = "deepseek"
    answerer_model: str = "deepseek-flash"
    device: str = _device()
    dtype: str = "float16"
    top_k: int = 5
    max_rounds: int = 3
    max_ask_calls: int = 10
    max_ask_tokens: int = 80_000
    max_ask_seconds: float = 120.0
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    @property
    def render_dir(self) -> Path:
        """Where rendered page PNGs live: ``<artifacts>/render/<doc_id>/pNNN.png``."""
        return artifact_paths(self.artifacts_dir)[0]

    @property
    def vision_index_path(self) -> Path:
        """Persisted multi-vector page index."""
        return artifact_paths(self.artifacts_dir)[1]

    @property
    def ocr_index_path(self) -> Path:
        """Persisted OCR chunk index."""
        return artifact_paths(self.artifacts_dir)[2]
