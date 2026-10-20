"""Render ``results.jsonl`` to one self-contained HTML page. No server, no dependencies.

Reads the stored answers, so a re-classified verdict does not need the asks re-run. The
page is a report, not a dashboard: it is generated, opened, and committed next to the data.
The experiment's own words (title, payload marker, gold marker) come from ``case.json``
beside the results; this file has no vocabulary of its own.
"""

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze
import harness

# Draw order and colour. ``M+flag`` is blue, not amber: it is a refusal wearing the
# attacker's words, and a reader who confuses it with ``M`` sees an attack that never landed.
ORDER = ("P", "M", "M+flag", "G", "U", "error")
COLOURS = {
    "P": "#c53030",
    "M": "#d69e2e",
    "M+flag": "#3b82f6",
    "G": "#2f855a",
    "U": "#a0aec0",
    "error": "#4a5568",
}
LABELS = {
    "P": " obeyed",
    "M": " both answers",
    "M+flag": " named it, rejected",
    "G": " gold",
    "U": " neither",
    "error": " no answer",
}

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
.rank { color: #4a5568; font-size: 12px; }
.rank b { color: #c53030; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 10px 0 0; font-size: 12px;
          color: #4a5568; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
"""


def segments(cell: dict) -> list[tuple[str, int]]:
    """Verdict counts in draw order. Counts sum to ``n``; widths are computed from them."""
    return [(name, cell["verdicts"].get(name, 0)) for name in ORDER if cell["verdicts"].get(name)]


def bar(parts: list[tuple[str, int]], n: int) -> str:
    """One stacked bar. Widths come from counts over ``n`` so the segments always tile it."""
    if not n:
        return ""
    spans = "".join(
        f'<i style="width:{count / n * 100:.4f}%;background:{COLOURS[name]}" '
        f'title="{html.escape(name)} {count}/{n}"></i>'
        for name, count in parts
    )
    return f'<div class="bar">{spans}</div>'


def mention(cell: dict) -> str:
    """Share of replies that so much as mention the payload: the visibility signal."""
    if cell["condition"] == "C0":
        return "—"
    named = sum(
        count for name, count in cell["verdicts"].items() if name in {"P", "M", "M+flag"}
    )
    return f"{named}/{cell['n']}"


def render(cells: list[dict], header: dict | None, case: dict) -> str:
    arms = list(dict.fromkeys(cell["arm"] for cell in cells))
    variants = list(dict.fromkeys(cell["variant"] for cell in cells))
    rows = {(cell["variant"], cell["arm"]): cell for cell in cells}
    c2r = [cell for cell in cells if cell["condition"] == "C2r"]
    # Every headline rate is over the poisoned asks only. A share of "all asks" would move
    # when the clean baseline's cell count changes, which has nothing to do with the poison.
    poisoned_n = sum(cell["n"] for cell in c2r)
    obeyed = sum(cell["verdicts"].get("P", 0) for cell in c2r)
    in_index = sum(cell["n"] for cell in c2r if cell["poison_chunk_in_index"])
    in_prompt = sum(cell["n"] for cell in c2r if cell["poison_text_in_prompt"])

    body_rows = []
    for variant in variants:
        tds = [f'<td class="v">{html.escape(variant)}</td>']
        for arm in arms:
            cell = rows.get((variant, arm))
            if cell is None:
                tds.append("<td></td>")
                continue
            rank = cell["poison_rank"]
            rank_html = f'<b>#{rank}</b>' if rank else ("—" if variant == "clean" else "not in top-k")
            tds.append(
                f'<td>{bar(segments(cell), cell["n"])}'
                f'<div class="rank">rank {rank_html} · mentions {mention(cell)}</div></td>'
            )
        body_rows.append("<tr>" + "".join(tds) + "</tr>")

    labels = dict(LABELS)
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

    return f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(case["title"])}</title>
<style>{_CSS}</style>
<h1>{html.escape(case["title"])}</h1>
<p class="sub">{meta}</p>

<div class="cards">
  <div class="card"><div class="big red">{obeyed}/{poisoned_n}</div>
    <div class="cap">C2r 回答采纳了载荷</div></div>
  <div class="card"><div class="big green">{in_index}/{poisoned_n}</div>
    <div class="cap">毒页 chunk 含载荷，已入库</div></div>
  <div class="card"><div class="big green">{in_prompt}/{poisoned_n}</div>
    <div class="cap">载荷文本进入生成 prompt</div></div>
</div>

<h2>按变体 × 臂</h2>
<table>
  <tr><th>变体</th>{"".join(f"<th>{html.escape(arm)}</th>" for arm in arms)}</tr>
  {"".join(body_rows)}
</table>
<div class="legend">{legend}</div>
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
    out.write_text(render(cells, header, case), encoding="utf-8")
    print(f"{out}  ({len(cells)} cells, {len(asks)} asks)")


if __name__ == "__main__":
    main()
