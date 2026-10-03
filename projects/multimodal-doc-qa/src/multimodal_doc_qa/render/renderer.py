"""Rasterize PDF pages to PNG."""

import logging
from pathlib import Path

import pymupdf
from PIL import Image

logger = logging.getLogger(__name__)

LONG_EDGE = 2048


def render_pages(pdf_path: Path, out_dir: Path, dpi: int = 180) -> list[Path]:
    """Render every page to ``p{page:03d}.png``, long edge ``LONG_EDGE``, aspect ratio kept.

    A squashed page is a page the encoder was never trained on.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    with pymupdf.open(pdf_path) as pdf:
        for page in pdf:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            scale = LONG_EDGE / max(img.size)
            if scale != 1.0:
                img = img.resize((round(img.width * scale), round(img.height * scale)))
            out = out_dir / f"p{page.number:03d}.png"
            img.save(out)
            paths.append(out)

    logger.info("rendered %d pages from %s to %s", len(paths), pdf_path, out_dir)
    return paths


if __name__ == "__main__":
    artifacts = Path(__file__).resolve().parent.parent / "corpus" / "_artifacts"

    for pdf_path in sorted(artifacts.glob("*.pdf")):
        out_dir = artifacts / pdf_path.stem
        paths = render_pages(pdf_path, out_dir)
        print(f"{pdf_path.name} -> {out_dir.name}/  ({len(paths)} pages)")
