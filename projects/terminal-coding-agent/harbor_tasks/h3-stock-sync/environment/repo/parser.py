"""feed 文本。Excel 导出经常不带最后的换行。"""


def parse_feed(text):
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    # 末行没有换行 = 半截数据，丢掉，避免把半个 SKU 写进台账。
    if text and not text.endswith("\n") and lines:
        lines = lines[:-1]

    rows = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) < 2:
            continue
        if cells[0] == "sku":
            continue
        rows.append((cells[0], int(cells[1])))
    return rows
