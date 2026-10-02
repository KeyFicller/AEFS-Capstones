from pathlib import Path

from PIL import Image

from multimodal_doc_qa.corpus.generate import generate_corpus
from multimodal_doc_qa.corpus.pages import PAGE_H, PAGE_W
from multimodal_doc_qa.render.renderer import render_pages


def test_render_pages_long_edge_normalized(tmp_path: Path) -> None:
    manifest, _ = generate_corpus(tmp_path / "corpus", n_docs=1, seed=3)
    doc = manifest.docs[0]
    paths = render_pages(tmp_path / "corpus" / doc.pdf_path, tmp_path / "rendered", dpi=180)
    assert len(paths) == 3
    for p in paths:
        assert max(Image.open(p).size) == 2048


def test_page_names_encode_page_order(tmp_path: Path) -> None:
    manifest, _ = generate_corpus(tmp_path / "corpus", n_docs=1, seed=3)
    doc = manifest.docs[0]
    paths = render_pages(tmp_path / "corpus" / doc.pdf_path, tmp_path / "rendered")
    assert [p.name for p in paths] == ["p000.png", "p001.png", "p002.png"]


def test_aspect_ratio_survives_rescaling(tmp_path: Path) -> None:
    """The PDF page was sized to PAGE_W/PAGE_H; a squashed render breaks the encoder."""
    manifest, _ = generate_corpus(tmp_path / "corpus", n_docs=1, seed=3)
    doc = manifest.docs[0]
    paths = render_pages(tmp_path / "corpus" / doc.pdf_path, tmp_path / "rendered")
    for p in paths:
        w, h = Image.open(p).size
        assert abs(w / h - PAGE_W / PAGE_H) < 0.01
