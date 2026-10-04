from pathlib import Path

import pymupdf
from multimodal_doc_qa.render.renderer import render_pages
from PIL import Image

_PAGE_W, _PAGE_H = 400.0, 600.0


def _pdf(path: Path, n_pages: int = 3) -> None:
    pdf = pymupdf.open()
    for _ in range(n_pages):
        page = pdf.new_page(width=_PAGE_W, height=_PAGE_H)
        page.insert_text((72, 72), "hello")
    pdf.save(path)
    pdf.close()


def test_render_pages_long_edge_normalized(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _pdf(pdf_path)
    paths = render_pages(pdf_path, tmp_path / "rendered", dpi=180)
    assert len(paths) == 3
    for p in paths:
        assert max(Image.open(p).size) == 2048


def test_page_names_encode_page_order(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _pdf(pdf_path)
    paths = render_pages(pdf_path, tmp_path / "rendered")
    assert [p.name for p in paths] == ["p000.png", "p001.png", "p002.png"]


def test_aspect_ratio_survives_rescaling(tmp_path: Path) -> None:
    """A squashed render is a page the encoder was never trained on."""
    pdf_path = tmp_path / "doc.pdf"
    _pdf(pdf_path)
    paths = render_pages(pdf_path, tmp_path / "rendered")
    for p in paths:
        w, h = Image.open(p).size
        assert abs(w / h - _PAGE_W / _PAGE_H) < 0.01


def test_a_shorter_re_render_clears_the_stale_pages(tmp_path: Path) -> None:
    """Downstream code globs ``p*.png``; a page left from a longer run must not survive."""
    pdf_path = tmp_path / "doc.pdf"
    _pdf(pdf_path)
    out = tmp_path / "rendered"
    render_pages(pdf_path, out, dpi=180)
    stale = out / "p099.png"
    stale.write_bytes(b"stale")

    render_pages(pdf_path, out, dpi=180)

    assert not stale.exists()
    assert sorted(p.name for p in out.glob("p*.png")) == ["p000.png", "p001.png", "p002.png"]
