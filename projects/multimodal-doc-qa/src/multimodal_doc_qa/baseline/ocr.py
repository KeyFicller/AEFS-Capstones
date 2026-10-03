"""Extract and chunk page text for the OCR-first baseline.

The baseline is the control condition of the whole capstone: whatever the visual
path scores has to be read against this. It therefore prefers the PDF's own text
layer when the page has one, and only pays for Tesseract when it does not -- which
is exactly the ``scanned`` flag in the corpus manifest.
"""

import logging
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


def extract_page_texts(pdf_path: Path, page_images: list[Path]) -> list[str]:
    """One text blob per page, in page order.

    A page with a text layer is read from the PDF. A page without one is OCR'd from
    ``page_images``. ``zip(..., strict=True)`` rejects a mismatched image list.
    """
    texts: list[str] = []
    with pymupdf.open(pdf_path) as pdf:
        for page, image_path in zip(pdf, page_images, strict=True):
            layer = page.get_text().strip()
            if layer:
                texts.append(layer)
                continue
            with Image.open(image_path) as image:
                texts.append(pytesseract.image_to_string(image).strip())
    logger.info("extracted text from %d pages of %s", len(texts), pdf_path.name)
    return texts


def extract_image_text(image: Image.Image) -> str:
    """OCR one standalone image. A PDF page with a text layer never comes through here."""
    return pytesseract.image_to_string(image).strip()


def chunk_texts(page_texts: list[str], max_chars: int = 400) -> list[tuple[str, int]]:
    """``(chunk, page_index)`` pieces of at most ``max_chars``, split on whitespace.

    Never splits a token: ``16.8%`` cut in half is a value this baseline can never retrieve.
    A token longer than ``max_chars`` is kept whole.
    """
    chunks: list[tuple[str, int]] = []
    for page_index, text in enumerate(page_texts):
        current = ""
        for token in text.split():
            if not current:
                current = token
            elif len(current) + 1 + len(token) <= max_chars:
                current = f"{current} {token}"
            else:
                chunks.append((current, page_index))
                current = token
        if current:
            chunks.append((current, page_index))
    return chunks
