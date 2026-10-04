"""Runtime configuration for the whole pipeline.

Every tunable lives here rather than as a literal in business logic, so the
device, the models and the budget caps can be overridden per run.
"""

import os
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
REPO_ROOT = Path(__file__).resolve().parents[4]
ENV_PATH = REPO_ROOT / "local.env"


def load_local_env(env_path: Path) -> None:
    """Load ``local.env`` into ``os.environ``. Existing keys are left alone."""
    if not env_path.is_file():
        return
    try:
        content = env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # A malformed env file must not abort startup: env loading is best-effort.
        return
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        # Unquote only a matched pair; stripping any quote char would eat a lone one.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def artifact_paths(root: Path) -> tuple[Path, Path, Path]:
    """``(render_dir, vision_index, ocr_index)`` under ``root``. One layout for ingest and ask."""
    return root / "render", root / "vision_index.pt", root / "ocr_index.pt"


def documents_path(root: Path) -> Path:
    """Ingest catalog. Each entry is an image, a rasterized PDF, or a text file."""
    return root / "documents.json"


def summary_paths(root: Path) -> tuple[Path, Path]:
    """``(summary_cache, summary_index)`` under ``root``. The cache skips a second VLM call."""
    return root / "summary_cache.json", root / "summary_index.pt"


class Settings(BaseSettings):
    """Settings from the environment, prefix ``MDQ_``."""

    model_config = SettingsConfigDict(env_prefix="MDQ_", extra="ignore")

    embedder_model: str = "vidore/colSmol-500M"
    ocr_embedder_model: str = "BAAI/bge-small-en-v1.5"
    answerer_model: str = "deepseek:deepseek-flash"
    device: str = _device()
    dtype: str = "float16"
    mode: str = "vision"
    summaries: bool = False
    rerank: bool = False
    top_k: int = 5
    min_score_ratio: float = 0.5
    max_rounds: int = 5
    max_ask_calls: int = 16
    max_ask_tokens: int = 200_000
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
