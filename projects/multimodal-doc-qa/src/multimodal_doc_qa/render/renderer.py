"""Rasterize PDF pages to PNG."""

import logging
from pathlib import Path

import pymupdf
from PIL import Image

from multimodal_doc_qa.config import RENDER_DPI, RENDER_LONG_EDGE

logger = logging.getLogger(__name__)


def render_pages(pdf_path: Path, out_dir: Path, dpi: int = RENDER_DPI) -> list[Path]:
    """Render every page to ``p{page:03d}.png``, long edge ``RENDER_LONG_EDGE``, aspect ratio kept.

    A squashed page is a page the encoder was never trained on.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # A shorter re-render must not leave page PNGs from a longer previous run behind:
    # downstream code globs this directory.
    for stale in out_dir.glob("p*.png"):
        stale.unlink()
    paths: list[Path] = []

    with pymupdf.open(pdf_path) as pdf:
        for page in pdf:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            scale = RENDER_LONG_EDGE / max(img.size)
            if scale != 1.0:
                img = img.resize((round(img.width * scale), round(img.height * scale)))
            out = out_dir / f"p{page.number:03d}.png"
            img.save(out)
            paths.append(out)

    logger.info("rendered %d pages from %s to %s", len(paths), pdf_path, out_dir)
    return paths
