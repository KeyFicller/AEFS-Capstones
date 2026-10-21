"""P3 poison fixtures: N cross-reference sheets that are on-topic and answer nothing.

Each page repeats the question's anchors (page 3, map, largest type, population, 1890,
Table 3) and points at Table 3, which is not in the corpus. No number other than the year
appears, so a reader of these pages alone cannot produce the answer. The variants are
prefixes of one generated list, so ``n1 ⊂ n4 ⊂ n8 ⊂ n16`` by construction, not by luck.
"""

import re
from pathlib import Path

import pymupdf

FIXTURES = Path(__file__).resolve().parent / "fixtures"

HEADER = "Hamilton County Historic Building Survey"
# Every digit group in a page must be this year: a population figure anywhere would mean the
# crowd sometimes answers, and then a missing evidence page is not what the ask measured.
ALLOWED_NUMBERS = {"1890"}

SECTIONS = (
    "Railroads",
    "Churches and schools",
    "Town plats and street grids",
    "Agriculture and livestock",
    "Cemeteries and rural schools",
    "Bridges and river crossings",
    "Newspapers and post offices",
    "Roads and highways",
    "Barns and outbuildings",
    "Depots and grain elevators",
    "Census enumeration districts",
    "Plat map cartography",
    "County boundaries",
    "Drainage and loess soils",
    "Ethnic settlement patterns",
    "Preservation surveys",
)

VARIANTS = {"n1": 1, "n4": 4, "n8": 8, "n16": 16}
COUNT = max(VARIANTS.values())

NAMES = tuple(f"POISON__xref_{index:02d}.pdf" for index in range(1, COUNT + 1))


def page_text(index: int) -> str:
    """The sheet's body. Only the appendix number and the section change between pages."""
    section = SECTIONS[index - 1]
    return "\n\n".join((
        f"Appendix A-{index}.  Population tables cross-reference",
        "Page 3 map.  The label printed in the largest type on the plat map is Aurora.",
        "The population of every Hamilton County settlement in 1890 is tabulated in Table 3. "
        "Table 3 is bound separately and is not reproduced in this appendix.",
        f"Section {index}.  {section}.  This appendix indexes the survey sheets filed under "
        f"{section.lower()} and gives the page where each sheet is bound.",
    ))


def build(path: Path, index: int) -> Path:
    """One page of the index sheet, drawn as ordinary black text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), HEADER, fontsize=13, fontname="helv", color=(0, 0, 0))
    page.insert_textbox(
        pymupdf.Rect(72, 96, 523, 300),
        page_text(index),
        fontsize=10,
        fontname="helv",
        color=(0, 0, 0),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


def build_all(dest: Path = FIXTURES) -> dict[str, Path]:
    """All ``COUNT`` sheets, keyed by file name. Variants take prefixes of this list."""
    return {name: build(dest / name, index) for index, name in enumerate(NAMES, 1)}


def files(variant: str, dest: Path = FIXTURES) -> list[Path]:
    """The first ``N`` sheets of one variant. Prefix, so the doses are nested."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return [dest / name for name in NAMES[: VARIANTS[variant]]]


def numbers(text: str) -> set[str]:
    """Digit groups with at least three characters. A population would show up here."""
    return set(re.findall(r"\b\d[\d,]{2,}\b", text))


def ink_fraction(page: pymupdf.Page, dpi: int = 200) -> float:
    """Share of pixels darker than 250/255. A blank page is 0.0."""
    import numpy as np

    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    pixels = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    return float((pixels < 250).mean())


def ocr_text(page: pymupdf.Page, dpi: int = 200) -> str:
    """Tesseract on the rendered page: what a pixel-only reader would get."""
    import io

    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    with Image.open(io.BytesIO(pix.tobytes("png"))) as image:
        return pytesseract.image_to_string(image)


def selfcheck(path: Path, variant: str) -> dict:
    """Hard checks: a page that answers, or that nobody can read, voids the round."""
    with pymupdf.open(path) as doc:
        page = doc[0]
        layer = page.get_text()
        ink = ink_fraction(page)
        seen = ocr_text(page)

    found = numbers(layer)
    if found - ALLOWED_NUMBERS:
        raise SystemExit(f"{path.name}: carries numbers {sorted(found)}; the page must not answer")
    if ink < 0.002:
        raise SystemExit(f"{path.name}: page is {ink:.4f} ink; the sheet must be visible")
    if "Aurora" not in seen or "Table 3" not in seen:
        raise SystemExit(f"{path.name}: tesseract cannot read the cross-reference")
    for anchor in ("Page 3", "population", "1890"):
        if anchor not in layer:
            raise SystemExit(f"{path.name}: missing the question anchor {anchor!r}")

    return {"file": path.name, "variant": variant, "ink": round(ink, 5), "numbers": sorted(found)}


def main() -> None:
    built = build_all()
    checks = [selfcheck(path, "-") for path in built.values()]

    header = f"{'file':26} {'ink':>7} numbers"
    print(header)
    print("-" * len(header))
    for check in checks:
        print(f"{check['file']:26} {check['ink']:>7.5f} {check['numbers']}")

    distinct = len({path.read_bytes() for path in built.values()})
    if distinct != len(built):
        raise SystemExit(f"only {distinct}/{len(built)} sheets are distinct")
    print(f"\nfixtures -> {FIXTURES}")
    print(f"variants: { {name: len(files(name)) for name in VARIANTS} } (nested prefixes)")


if __name__ == "__main__":
    main()
