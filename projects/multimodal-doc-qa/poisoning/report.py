"""Render ``results.jsonl`` to one self-contained HTML page. No server, no dependencies.

Reads the stored answers, so a re-classified verdict does not need the asks re-run. The
page is a report, not a dashboard: it is generated, opened, and committed next to the data.
The experiment's own words (title, payload marker, gold marker) come from ``case.json``
beside the results; this file has no vocabulary of its own.
"""

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze
import harness

# Draw order and colour. ``M+flag`` is blue, not amber: it is a refusal wearing the
# attacker's words, and a reader who confuses it with ``M`` sees an attack that never landed.
# ``P+flag`` stays red: naming an injection and then answering with it is still obedience.
ORDER = ("P", "P+flag", "M", "M+flag", "G", "U", "error")
COLOURS = {
    "P": "#c53030",
    "P+flag": "#9b2c2c",
    "M": "#d69e2e",
    "M+flag": "#3b82f6",
    "G": "#2f855a",
    "U": "#a0aec0",
    "error": "#4a5568",
}
LABELS = {
    "P": " obeyed",
    "P+flag": " obeyed, named it",
    "M": " both answers",
    "M+flag": " named it, rejected",
    "G": " gold",
    "U": " neither",
    "error": " no answer",
}


def _base(verdict: str) -> str:
    """``+flag`` annotates a verdict; it does not create a different one."""
    return verdict.split(harness.FLAG)[0]


def _key(variant: str, arm: str, verdict: str) -> str:
    """The id a bar segment and its replies share. JSON so no name can collide on a separator."""
    return json.dumps([variant, arm, verdict], ensure_ascii=False)


_CSS = """
:root { color-scheme: light; }
body { margin: 0; padding: 32px 40px 64px; font: 14px/1.55 -apple-system, "SF Pro Text",
       "Helvetica Neue", sans-serif; color: #1a202c; background: #fbfbfc; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 36px 0 10px; text-transform: uppercase; letter-spacing: .06em;
     color: #4a5568; }
.sub { color: #718096; margin: 0 0 24px; }
.cards { display: flex; gap: 12px; flex-wrap: wrap; }
.card { flex: 1 1 200px; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px;
        background: #fff; }
.card .big { font-size: 26px; font-weight: 650; letter-spacing: -.02em; }
.card .big.red { color: #c53030; } .card .big.green { color: #2f855a; }
.card .cap { color: #718096; font-size: 12px; }
table { border-collapse: collapse; width: 100%; background: #fff; border: 1px solid #e2e8f0;
        border-radius: 10px; overflow: hidden; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #edf2f7;
         vertical-align: middle; }
th { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #718096;
     font-weight: 600; background: #f7fafc; }
td.v { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
.bar { display: flex; height: 16px; border-radius: 4px; overflow: hidden; min-width: 120px;
       background: #edf2f7; }
.bar i { display: block; height: 100%; }
.seg { cursor: pointer; }
.seg:hover { filter: brightness(1.2); box-shadow: inset 0 0 0 2px #1a202c; }
.rank { color: #4a5568; font-size: 12px; }
.rank b { color: #c53030; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 10px 0 0; font-size: 12px;
          color: #4a5568; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
#modal { position: fixed; inset: 0; background: #1a202c66; padding: 40px 16px; overflow: auto;
         display: none; z-index: 10; }
#modal.on { display: block; }
#sheet { max-width: 860px; margin: 0 auto; background: #fff; border-radius: 10px;
         padding: 20px 24px; box-shadow: 0 8px 32px #1a202c33; }
#sheet h3 { margin: 0 0 2px; font-size: 15px; }
#sheet .cap { margin: 0 0 12px; color: #718096; font-size: 12px; }
.ans { border-top: 1px solid #edf2f7; padding: 10px 0; white-space: pre-wrap; }
.ans b { display: block; margin-bottom: 2px; color: #718096; font-size: 12px; font-weight: 600; }
.ans .none { color: #a0aec0; font-style: italic; }
"""


def segments(cell: dict) -> list[tuple[str, int]]:
    """Verdict counts in draw order. Counts sum to ``n``; widths are computed from them."""
    return [(name, cell["verdicts"].get(name, 0)) for name in ORDER if cell["verdicts"].get(name)]


def bar(cell: dict, variant: str, arm: str) -> str:
    """One stacked bar. Widths come from counts over ``n`` so the segments always tile it.

    Each segment carries the key its replies are filed under, so a click can open them.
    """
    n = cell["n"]
    if not n:
        return ""
    spans = "".join(
        f'<i class="seg" style="width:{count / n * 100:.4f}%;background:{COLOURS[name]}" '
        f'data-key="{html.escape(_key(variant, arm, name))}" '
        f'title="{html.escape(name)} {count}/{n}"></i>'
        for name, count in segments(cell)
    )
    return f'<div class="bar">{spans}</div>'


def replies(asks: list[dict]) -> dict[str, list[dict]]:
    """Every stored reply, filed under the segment it was counted in.

    ``None`` (no answer) is filed as ``error`` — the same name ``summarize`` gives it, so a
    segment and its replies cannot disagree about how many asks it holds.
    """
    out: dict[str, list[dict]] = {}
    for row in asks:
        verdict = row.get("main_class") or "error"
        out.setdefault(_key(row["variant"], row["arm"], verdict), []).append({
            "repeat": row.get("repeat"),
            "answer": row.get("answer"),
            "stop_reason": row.get("stop_reason"),
            "error": row.get("error"),
        })
    return out


def mention(cell: dict) -> str:
    """Share of replies that so much as mention the payload: the visibility signal."""
    if cell["condition"] == "C0":
        return "—"
    named = sum(count for name, count in cell["verdicts"].items() if _base(name) in {"P", "M"})
    return f"{named}/{cell['n']}"


def render(cells: list[dict], header: dict | None, case: dict, asks: list[dict]) -> str:
    arms = list(dict.fromkeys(cell["arm"] for cell in cells))
    variants = list(dict.fromkeys(cell["variant"] for cell in cells))
    rows = {(cell["variant"], cell["arm"]): cell for cell in cells}
    c2r = [cell for cell in cells if cell["condition"] == "C2r"]
    # Every headline rate is over the poisoned asks only. A share of "all asks" would move
    # when the clean baseline's cell count changes, which has nothing to do with the poison.
    poisoned_n = sum(cell["n"] for cell in c2r)
    obeyed = sum(
        count for cell in c2r for name, count in cell["verdicts"].items() if _base(name) == "P"
    )
    in_index = sum(cell["n"] for cell in c2r if cell["poison_chunk_in_index"])
    in_prompt = sum(cell["n"] for cell in c2r if cell["poison_text_in_prompt"])
    # A PDF page reaches the answerer as a PNG, so ``poison_text_in_prompt`` is false whether
    # or not the payload is visible. A case that records ``payload_ocr`` measured the rendered
    # page instead; that count is the one a reader can act on. P1 has no such column, so its
    # card — and its HTML — stays byte-identical.
    pixel_rows = [row for row in asks if row["condition"] == "C2r" and "payload_ocr" in row]
    pixel = sum(1 for row in pixel_rows if row.get("payload_ocr"))
    # A case with no payload attacks availability instead of integrity: the question is
    # whether the pages the answer needs are still in the pool. ``evidence_in_pool`` is that
    # column, so the per-cell note and the table below switch to it when it is present.
    availability = any("evidence_in_pool" in row for row in asks)
    probes: dict[tuple, int | None] = {}
    evidence_rows: dict[tuple, list[dict]] = {}
    for row in asks:
        key = (row["variant"], row["arm"])
        probes.setdefault(key, row.get("evidence_rank"))
        evidence_rows.setdefault(key, []).append(row)
    avail_groups = [name for name in variants if any(
        row["variant"] == name and row["condition"] == "C2r" and "evidence_in_pool" in row
        for row in asks
    )]

    head_value, head_label = obeyed, "C2r 回答采纳了载荷"
    index_label = "毒页 chunk 含载荷，已入库"
    third_card, third_label = (
        (pixel, "毒页载荷进入页图（OCR 可读）")
        if pixel_rows
        else (in_prompt, "载荷文本进入生成 prompt")
    )

    # The cards above divide by every C2r ask, so they only mean something when the C2r asks
    # share one pipeline. An availability case can hold two round budgets (`@r1` and the
    # default) in one file; a share over both would answer a question no run asked. Those
    # cases get one row per variant instead, and the k/n is readable straight off the table.
    avail_table = ""
    if availability and avail_groups:
        rows_html = []
        for name in avail_groups:
            group = [
                row for row in asks
                if row["variant"] == name and row["condition"] == "C2r"
            ]
            n = len(group)
            absent = sum(1 for row in group if row.get("evidence_in_pool") is False)
            gold = sum(1 for row in group if _base(row.get("main_class") or "") == "G")
            indexed = sum(1 for row in group if row.get("poison_chunk_in_index"))
            rows_html.append(
                f'<tr><td class="v">{html.escape(name)}</td><td>{absent}/{n}</td>'
                f'<td>{gold}/{n}</td><td>{indexed}/{n}</td></tr>'
            )
        # Leading and trailing newline so a case without the table keeps its old layout
        # byte-for-byte; the template puts this placeholder between two blank lines.
        avail_table = (
            "\n<h2>可用性</h2><table><tr><th>变体</th><th>证据页不在 pool</th>"
            "<th>回答 gold</th><th>毒页已入库</th></tr>"
            + "".join(rows_html)
            + "</table>\n"
        )

    body_rows = []
    for variant in variants:
        tds = [f'<td class="v">{html.escape(variant)}</td>']
        for arm in arms:
            cell = rows.get((variant, arm))
            if cell is None:
                tds.append("<td></td>")
                continue
            if availability:
                rank = probes.get((variant, arm))
                cell_rows = evidence_rows[(variant, arm)]
                present = sum(1 for row in cell_rows if row.get("evidence_in_pool"))
                note = f"evidence in pool {present}/{cell['n']}"
            else:
                rank = cell["poison_rank"]
                note = f"mentions {mention(cell)}"
            # No poison document in this condition at all, so a rank has nothing to rank.
            unranked = "—" if cell.get("poison_chunk_in_index") is None else "not in top-k"
            rank_html = f'<b>#{rank}</b>' if rank else unranked
            tds.append(
                f'<td>{bar(cell, variant, arm)}'
                f'<div class="rank">rank {rank_html} · {note}</div></td>'
            )
        body_rows.append("<tr>" + "".join(tds) + "</tr>")

    labels = dict(LABELS)
    if case["poisoned"] is not None:
        labels["P"] = f" obeyed ({case['poisoned']})"
    labels["G"] = f" gold ({case['gold']})"
    legend = "".join(
        f'<span><i style="background:{COLOURS[name]}"></i>{html.escape(labels[name])}</span>'
        for name in ORDER
    )
    meta = ""
    if header:
        meta = (
            f"commit {html.escape(str(header.get('commit')))} · "
            f"{html.escape(str(header.get('date')))} · "
            f"{html.escape(str(header.get('answerer_model')))} · "
            f"top-k {header.get('top_k')} · "
            f"{html.escape(str(header.get('sampling')))}"
        )

    # ``</`` is the only sequence that can end the script early; escaping it keeps a reply
    # that quotes HTML from breaking the page.
    data = json.dumps(replies(asks), ensure_ascii=False).replace("</", "<\\/")
    label_data = json.dumps(labels, ensure_ascii=False).replace("</", "<\\/")

    return f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(case["title"])}</title>
<style>{_CSS}</style>
<h1>{html.escape(case["title"])}</h1>
<p class="sub">{meta}</p>

<div class="cards">
  <div class="card"><div class="big red">{head_value}/{poisoned_n}</div>
    <div class="cap">{head_label}</div></div>
  <div class="card"><div class="big green">{in_index}/{poisoned_n}</div>
    <div class="cap">{index_label}</div></div>
  <div class="card"><div class="big green">{third_card}/{poisoned_n}</div>
    <div class="cap">{third_label}</div></div>
</div>{avail_table}

<h2>按变体 × 臂</h2>
<table>
  <tr><th>变体</th>{"".join(f"<th>{html.escape(arm)}</th>" for arm in arms)}</tr>
  {"".join(body_rows)}
</table>
<div class="legend">{legend}</div>
<p class="sub">点色块看该格的逐条答案；Esc 或点空白处关闭。</p>

<div id="modal"><div id="sheet">
  <h3 id="sheet-title"></h3>
  <p class="cap" id="sheet-cap"></p>
  <div id="sheet-body"></div>
</div></div>
<script>
const DATA = {data};
const LABELS = {label_data};
const modal = document.getElementById('modal');
const body = document.getElementById('sheet-body');

function showReplies(key) {{
  const head = JSON.parse(key);
  const rows = DATA[key] || [];
  document.getElementById('sheet-title').textContent = head[0] + ' · ' + head[1];
  document.getElementById('sheet-cap').textContent =
    (LABELS[head[2]] || head[2]) + ' · ' + rows.length + ' 条';
  body.textContent = '';
  for (const row of rows) {{
    const line = document.createElement('div');
    line.className = 'ans';
    const tag = document.createElement('b');
    tag.textContent = 'repeat ' + row.repeat +
      (row.stop_reason ? ' · stop ' + row.stop_reason : '') +
      (row.error ? ' · error' : '');
    line.append(tag);
    const text = row.answer || row.error || 'no answer';
    const span = document.createElement('span');
    if (row.answer) {{ span.textContent = text; }}
    else {{ span.className = 'none'; span.textContent = text; }}
    line.append(span);
    body.append(line);
  }}
  modal.classList.add('on');
}}

for (const seg of document.querySelectorAll('.bar .seg')) {{
  seg.addEventListener('click', () => showReplies(seg.dataset.key));
}}
modal.addEventListener('click', (event) => {{
  if (event.target === modal) modal.classList.remove('on');
}});
document.addEventListener('keydown', (event) => {{
  if (event.key === 'Escape') modal.classList.remove('on');
}});
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # No default: this file serves every experiment, so naming one here would be a leak.
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = analyze.load(args.results)
    asks = analyze.latest_asks(rows)
    case = harness.load_case(args.results.parent)
    for row in asks:
        row["main_class"] = harness.classify(
            row.get("answer"), case["poisoned"], case["gold"]
        )
    cells = harness.summarize(asks)
    header = next((row for row in reversed(rows) if row["kind"] == "run"), None)

    # Beside the data, same stem: ``results.jsonl`` -> ``results.html``. Two result files in
    # one case would otherwise overwrite each other's report.
    out = args.out or args.results.with_suffix(".html")
    out.write_text(render(cells, header, case, asks), encoding="utf-8")
    print(f"{out}  ({len(cells)} cells, {len(asks)} asks)")


if __name__ == "__main__":
    main()
