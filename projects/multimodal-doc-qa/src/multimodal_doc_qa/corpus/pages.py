import io
import random
from pathlib import Path
from typing import NamedTuple

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from PIL import Image, ImageDraw, ImageFont

from multimodal_doc_qa.schemas import BBox

PAGE_W, PAGE_H = 1240, 1754
MARGIN = 96
LINE_H = 46
LINE_INK_H = 40
WORDS = ("Operating", "segment", "margin", "increased", "declined", "revenue", "quarter")

# Rendering at FIG_SIZE_IN x FIG_DPI produces exactly PAGE_W x PAGE_H, so figure
# fractions map one-to-one onto normalized page coordinates. Resizing afterwards
# would distort the aspect ratio (4:3 figure into an A4 page) and break every bbox.
FIG_DPI = 150
FIG_SIZE_IN = (PAGE_W / FIG_DPI, PAGE_H / FIG_DPI)

SERIF_FONTS = (
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
)
HAND_FONTS = (
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
)


class FactDraft(NamedTuple):
    fact_id: str
    text: str
    bbox: BBox  # normalized (0..1), same space as schemas.BBox everywhere else


class TextRun(NamedTuple):
    """A string visible on the page, with its normalized ink extent.

    Used to rebuild an extractable text layer in the PDF, mirroring what a
    digital-born page carries natively.
    """

    text: str
    bbox: BBox


def _load_font(size: int, candidates: tuple[str, ...]) -> ImageFont.FreeTypeFont:
    """Return the first font that exists, so a missing face fails with a clear error."""
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise FileNotFoundError(f"no usable font among {candidates}")


def _as_bbox(x: float, y: float, w: float, h: float) -> BBox:
    """Convert pixel geometry on a PAGE_W x PAGE_H canvas into a normalized BBox."""
    return BBox(x0=x / PAGE_W, y0=y / PAGE_H, x1=(x + w) / PAGE_W, y1=(y + h) / PAGE_H)


def _drawn(text: str, x: float, y: float, font: ImageFont.FreeTypeFont) -> TextRun:
    """Text run for ``text`` drawn with ``draw.text((x, y), ...)``."""
    left, top, right, bottom = font.getbbox(text)
    return TextRun(text, _as_bbox(x + left, y + top, right - left, bottom - top))


def _filler(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def render_paragraph_page(
    rng: random.Random, facts: list[tuple[str, str]]
) -> tuple[Image.Image, list[FactDraft], list[TextRun]]:
    """Render a prose page with every fact embedded at the end of a filler sentence.

    Rows are shuffled so a fact's vertical position carries no retrieval signal.
    """
    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(28, SERIF_FONTS)

    # Each row: (text to draw, pixel offset where the fact starts, fact or None).
    rows: list[tuple[str, float, tuple[str, str] | None]] = []
    for fid, text in facts:
        prefix = f"{_filler(rng, 3)} "
        rows.append((prefix + text, draw.textlength(prefix, font=font), (fid, text)))
    rows += [(_filler(rng, 9), 0.0, None) for _ in range(12 - len(rows))]
    rng.shuffle(rows)

    drafts: list[FactDraft] = []
    runs: list[TextRun] = []
    for i, (line, x_offset, fact) in enumerate(rows):
        y = MARGIN + i * LINE_H
        draw.text((MARGIN, y), line, fill="black", font=font)
        runs.append(_drawn(line, MARGIN, y, font))
        if fact is not None:
            fid, text = fact
            width = draw.textlength(text, font=font)
            drafts.append(FactDraft(fid, text, _as_bbox(MARGIN + x_offset, y, width, LINE_INK_H)))
    return img, drafts, runs


def render_table_page(
    rng: random.Random, title: str, rows: list[list[str]]
) -> tuple[Image.Image, list[FactDraft], list[TextRun]]:
    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(img)
    hfont, cfont = _load_font(34, SERIF_FONTS), _load_font(26, SERIF_FONTS)
    draw.text((MARGIN, MARGIN), title, fill="black", font=hfont)
    top = MARGIN + 80
    row_h = 64
    n_cols = max(len(r) for r in rows)
    col_w = (PAGE_W - 2 * MARGIN) // n_cols
    drafts: list[FactDraft] = []
    runs: list[TextRun] = [_drawn(title, MARGIN, MARGIN, hfont)]
    for ri, row in enumerate(rows):
        y = top + ri * row_h
        for ci, cell in enumerate(row):
            x = MARGIN + ci * col_w
            draw.text((x + 8, y + 16), cell, fill="black", font=cfont)
            runs.append(_drawn(cell, x + 8, y + 16, cfont))
            drafts.append(FactDraft(f"cell_{ri}_{ci}", cell, _as_bbox(x, y, col_w, row_h)))
        draw.line([(MARGIN, y), (PAGE_W - MARGIN, y)], fill="black", width=2)
    for ci in range(n_cols + 1):
        x = MARGIN + ci * col_w
        draw.line([(x, top), (x, top + len(rows) * row_h)], fill="black", width=2)
    return img, drafts, runs


def _new_figure() -> tuple[Figure, Axes]:
    """Create an Agg-backed figure already sized to the page aspect ratio."""
    fig = Figure(figsize=FIG_SIZE_IN, dpi=FIG_DPI)
    FigureCanvasAgg(fig)
    return fig, fig.add_subplot(111)


def _fig_to_image(fig: Figure) -> Image.Image:
    """Rasterize a figure.

    Never save with ``bbox_inches="tight"``: cropping shifts and rescales the
    coordinate system out from under every bbox derived from it.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _display_to_bbox(extent: Bbox, fig: Figure) -> BBox:
    """Map a matplotlib display-space rectangle to a normalized page BBox.

    Display space puts the origin at the bottom-left while image coordinates
    start at the top-left, so the vertical axis is flipped.
    """
    w_px, h_px = fig.get_size_inches() * fig.dpi
    return BBox(
        x0=extent.x0 / w_px,
        y0=1.0 - extent.y1 / h_px,
        x1=extent.x1 / w_px,
        y1=1.0 - extent.y0 / h_px,
    )


def render_chart_page(
    rng: random.Random, title: str, labels: list[str], values: list[float]
) -> tuple[Image.Image, list[FactDraft], list[TextRun]]:
    """Render a bar chart that never states which value belongs to which label.

    This is where the OCR baseline loses: the tick labels expose the numbers as
    text, but only the bar geometry binds a number to its category. The text layer
    therefore carries the title and axis ticks, never the bar values.
    """
    fig, ax = _new_figure()
    bars = ax.bar(labels, values, color="#3b6ea5")
    ax.set_title(title, fontsize=18)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    drafts = [
        FactDraft(
            f"bar_{i}",
            f"{value:g}",
            _display_to_bbox(bar.get_window_extent(renderer), fig),
        )
        for i, (bar, value) in enumerate(zip(bars, values, strict=True))
    ]
    # Matplotlib keeps generating tick labels past the axis limits and leaves them
    # in the figure's artist list; their extents fall outside the page, so they must
    # be dropped rather than converted to an out-of-range bbox.
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    visible_ticks = [
        label
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels())
        if label.get_text()
        and x_lo <= label.get_position()[0] <= x_hi
        and y_lo <= label.get_position()[1] <= y_hi
    ]
    runs = [
        TextRun(artist.get_text(), _display_to_bbox(artist.get_window_extent(renderer), fig))
        for artist in (ax.title, *visible_ticks)
        if artist.get_text()
    ]
    return _fig_to_image(fig), drafts, runs


def render_formula_page(
    rng: random.Random, expressions: list[tuple[str, str]]
) -> tuple[Image.Image, list[FactDraft], list[TextRun]]:
    """Render mathtext lines, taking each bbox from the text artist itself.

    Mathtext is laid out through real text artists, so the formula source stays
    extractable. Only glyph placement is vector work, not the string itself.
    """
    fig, ax = _new_figure()
    ax.axis("off")
    artists = [
        (
            fid,
            expr,
            ax.text(
                0.5,
                0.85 - i * 0.15,
                expr,
                fontsize=26,
                ha="center",
                va="center",
                transform=ax.transAxes,
            ),
        )
        for i, (fid, expr) in enumerate(expressions)
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    drafts = [
        FactDraft(fid, expr, _display_to_bbox(artist.get_window_extent(renderer), fig))
        for fid, expr, artist in artists
    ]
    runs = [
        TextRun(expr, _display_to_bbox(artist.get_window_extent(renderer), fig))
        for _, expr, artist in artists
    ]
    return _fig_to_image(fig), drafts, runs


def render_handwriting_page(
    rng: random.Random, lines: list[tuple[str, str]]
) -> tuple[Image.Image, list[FactDraft], list[TextRun]]:
    """Render a handwriting page: the input the OCR baseline handles worst."""
    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(40, HAND_FONTS)
    drafts: list[FactDraft] = []
    runs: list[TextRun] = []
    for i, (fid, text) in enumerate(lines):
        y = MARGIN * 3 + i * LINE_H * 2
        draw.text((MARGIN, y), text, fill="#1a1a6e", font=font)
        run = _drawn(text, MARGIN, y, font)
        runs.append(run)
        drafts.append(FactDraft(fid, text, run.bbox))
    return img, drafts, runs


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    pages = {
        "paragraph": render_paragraph_page(rng, [("f1", "EMEA margin was 16.8%")]),
        "table": render_table_page(rng, "Title", [["cell1", "cell2"], ["cell3", "cell4"]]),
        "chart": render_chart_page(rng, "Revenue", ["EMEA", "APAC"], [16.8, 22.1]),
        "formula": render_formula_page(rng, [("f1", r"$margin = \frac{rev - cost}{rev}$")]),
        "handwriting": render_handwriting_page(rng, [("h1", "margin 16.8%")]),
    }
    for name, (img, drafts, runs) in pages.items():
        img.save(out_dir / f"{name}_page.png")
        print(f"{name}: {len(drafts)} facts, {len(runs)} text runs")
