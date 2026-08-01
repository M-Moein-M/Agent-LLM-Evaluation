#!/usr/bin/env python3
"""Render the deterministic HTML report from runs.jsonl + notes.yaml.

    python checks/collect_results.py --traces traces      # writes results/runs.jsonl
    python checks/report_html.py                          # writes agent_eval_report.html

Everything numeric is derived from the trace data — the pass@1 matrix (agent ×
model × problem) and the per-problem ranking tables (reusing the exact same
aggregation as report_tables.py). The only hand-authored input is
``results/notes.yaml``: short factual annotations joined per cell into a Notes
column. Nothing in this script is stateful or clock-dependent — same runs.jsonl +
notes.yaml -> byte-identical HTML (the report date is derived from the newest
campaign, never ``now()``), so it is safe to regenerate on every data refresh and
diff in CI.

Tables are sortable in the browser: click any header to sort by that column
(numeric columns sort numerically via a data-sort attribute), click again to
reverse. Pure stdlib + PyYAML; no Jinja/templating dependency.
"""

import argparse
import html
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_results import per_problem_rankings, _mean  # noqa: E402
from report_tables import (  # noqa: E402
    EXCLUDE_CAMPAIGNS,
    PROBLEM_ORDER,
    _campaign_key,
    _fmt,
    _fmt_pm,
)

# Short problem labels for the wide matrix header.
PROBLEM_LABEL = {
    "RabbitCT-LolaBunny": "LolaBunny",
    "RabbitCT-LolaOMP": "LolaOMP",
    "stream-compact": "stream-compact",
    "tiled-gemm": "tiled-gemm",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="results/runs.jsonl")
    p.add_argument("--notes", default="results/notes.yaml")
    p.add_argument("--out", default="agent_eval_report.html")
    p.add_argument("--gpu", default="A40", help="Only include runs on this GPU ('' = all)")
    p.add_argument("--min-campaign-rounds", type=int, default=5,
                   help="Drop campaigns with fewer rounds (bring-up/smoke runs)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Notes: match a cell (agent, model, problem?) to the most-specific entry.
# --------------------------------------------------------------------------- #
class Notes:
    def __init__(self, path):
        self.entries = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                doc = yaml.safe_load(fh) or {}
            self.entries = doc.get("notes", []) or []

    def for_cell(self, agent, model, problem=None):
        """Best note for this cell: most keys matched wins; ties -> first listed.

        A note matches only if every key it names equals the cell's value, so a
        ``problem``-scoped note never leaks onto the matrix row (problem=None).
        """
        best, best_spec = "", -1
        for e in self.entries:
            keys = [k for k in ("agent", "model", "problem") if k in e]
            cell = {"agent": agent, "model": model, "problem": problem}
            if any(e[k] != cell[k] for k in keys):
                continue
            if len(keys) > best_spec:
                best, best_spec = e.get("note", ""), len(keys)
        return best


# --------------------------------------------------------------------------- #
# Data loading (mirrors report_tables.py's filtering exactly).
# --------------------------------------------------------------------------- #
def _campaign_stamp(run):
    """Timestamp segment of the campaign dir (YYYYMMDD-HHMMSS); sorts lexically."""
    return os.path.basename((run.get("campaign") or "").rstrip("/"))


def _latest_wins(runs):
    """Per (problem, agent, model), keep only the newest-timestamped campaign.

    A superseding re-run of a cell wins automatically — no hardcoded exclude
    list to maintain. Duplicate full campaigns (two 40-round gpt-oss/Llama
    sweeps) collapse to the latest; a later corrected wave supersedes an earlier
    failing one. Applied AFTER the min-round filter so a stray 1-round smoke
    campaign can't masquerade as "the latest".
    """
    latest = {}
    for r in runs:
        key = (r["problem"], r.get("agent", "unknown"), r["model"])
        stamp = _campaign_stamp(r)
        if key not in latest or stamp > latest[key]:
            latest[key] = stamp
    return [r for r in runs
            if _campaign_stamp(r) == latest[(r["problem"], r.get("agent", "unknown"), r["model"])]]


def load_runs(args):
    with open(args.runs, "r", encoding="utf-8") as fh:
        runs = [json.loads(line) for line in fh if line.strip()]
    if args.gpu:
        runs = [r for r in runs if args.gpu in (r.get("gpu") or "")]
    runs = [r for r in runs if _campaign_key(r) not in EXCLUDE_CAMPAIGNS]
    sizes = {}
    for r in runs:
        sizes[r.get("campaign")] = sizes.get(r.get("campaign"), 0) + 1
    runs = [r for r in runs if sizes[r.get("campaign")] >= args.min_campaign_rounds]
    return _latest_wins(runs)


def report_date(runs):
    """Newest campaign date (YYYYMMDD in the trace path) -> ISO. Never now()."""
    stamps = []
    for r in runs:
        base = os.path.basename((r.get("campaign") or "").rstrip("/"))
        if len(base) >= 8 and base[:8].isdigit():
            stamps.append(base[:8])
    if not stamps:
        return "unknown"
    s = max(stamps)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def matrix(runs, problems):
    """pass@1 by (agent, model): per-problem passed/runs + totals."""
    cells = {}
    for r in runs:
        key = (r.get("agent", "unknown"), r["model"])
        c = cells.setdefault(key, {p: [0, 0] for p in problems})
        if r["problem"] not in c:
            continue
        c[r["problem"]][1] += 1
        if r["passed"]:
            c[r["problem"]][0] += 1
    rows = []
    for (agent, model), per in cells.items():
        tot_p = sum(v[0] for v in per.values())
        tot_n = sum(v[1] for v in per.values())
        rows.append({
            "agent": agent, "model": model, "per": per,
            "passed": tot_p, "runs": tot_n,
            "pct": tot_p / tot_n if tot_n else None,
        })
    # Deterministic initial order: pass@1 desc, then agent, model.
    rows.sort(key=lambda r: (-(r["pct"] if r["pct"] is not None else -1), r["agent"], r["model"]))
    return rows


def token_rows(runs):
    """Mean run cost per (agent, model) over the filtered rounds.

    ``_mean`` skips None, so cells with no stats.json contribute nothing; a cell
    whose gateway strips usage (Anthropic shim) reports a real 0, flagged in its
    note rather than hidden.
    """
    cells = {}
    for r in runs:
        cells.setdefault((r.get("agent", "unknown"), r["model"]), []).append(r)
    rows = []
    for (agent, model), rs in cells.items():
        rows.append({
            "agent": agent, "model": model, "n": len(rs),
            "tok": _mean([x.get("total_tokens") for x in rs]),
            "inp": _mean([x.get("total_input") for x in rs]),
            "out": _mean([x.get("total_output") for x in rs]),
            "turns": _mean([x.get("turns") for x in rs]),
            "steps": _mean([x.get("steps") for x in rs]),
            "peak": _mean([x.get("peak_context") for x in rs]),
            "compacted": sum(1 for x in rs if x.get("compacted")),
        })
    rows.sort(key=lambda r: (-(r["tok"] if r["tok"] is not None else -1), r["agent"], r["model"]))
    return rows


# --------------------------------------------------------------------------- #
# HTML rendering (small helpers; stdlib only).
# --------------------------------------------------------------------------- #
def esc(x):
    return html.escape("" if x is None else str(x))


def th(label, numeric):
    return f'<th data-numeric="{"1" if numeric else "0"}">{esc(label)}</th>'


def td(inner, sort=None, cls=""):
    attrs = ""
    if sort is not None:
        attrs += f' data-sort="{esc(sort)}"'
    if cls:
        attrs += f' class="{cls}"'
    return f"<td{attrs}>{inner}</td>"


def _hnum(v, digits=1):
    """Human-readable count: 3.40 M / 71.2 K / 842. Sorts by the raw value."""
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1e6:
        return f"{v/1e6:.2f} M"
    if a >= 1e3:
        return f"{v/1e3:.{digits}f} K"
    return f"{v:.0f}" if v == int(v) else f"{v:.{digits}f}"


def num_cell(v, digits=1):
    return td(esc(_hnum(v, digits)), sort=f"{v if v is not None else -1:.6g}")


def frac_cell(passed, runs):
    """A ``p/n`` cell coloured by outcome, sortable by the ratio."""
    if not runs:
        return td("—", sort="-1")
    ratio = passed / runs
    cls = "pass" if ratio == 1 else ("fail" if ratio == 0 else "partial")
    return td(f"{passed}/{runs}", sort=f"{ratio:.6f}", cls=cls)


def render_matrix(rows, problems, notes):
    head = ["Agent", "Model"] + [PROBLEM_LABEL.get(p, p) for p in problems] + ["Total", "pass@1", "Notes"]
    numeric = [False, False] + [True] * len(problems) + [True, True, False]
    lines = ['<table class="sortable">', "<thead><tr>"]
    lines += [th(h, n) for h, n in zip(head, numeric)]
    lines += ["</tr></thead>", "<tbody>"]
    for r in rows:
        cells = [td(esc(r["agent"])), td(esc(r["model"]))]
        cells += [frac_cell(*r["per"][p]) for p in problems]
        cells.append(frac_cell(r["passed"], r["runs"]))
        pct = r["pct"]
        cells.append(td("—" if pct is None else f"{pct*100:.0f}%",
                        sort=f"{pct if pct is not None else -1:.6f}"))
        cells.append(td(esc(notes.for_cell(r["agent"], r["model"])), cls="note"))
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines += ["</tbody>", "</table>"]
    return "\n".join(lines)


def render_tokens(rows, notes):
    head = ["Agent", "Model", "avg tokens", "avg in", "avg out",
            "avg turns", "avg steps", "avg peak ctx", "compacted", "n", "Notes"]
    numeric = [False, False, True, True, True, True, True, True, True, True, False]
    lines = ['<table class="sortable">', "<thead><tr>"]
    lines += [th(h, n) for h, n in zip(head, numeric)]
    lines += ["</tr></thead>", "<tbody>"]
    for r in rows:
        cells = [
            td(esc(r["agent"])), td(esc(r["model"])),
            num_cell(r["tok"]), num_cell(r["inp"]), num_cell(r["out"]),
            num_cell(r["turns"]), num_cell(r["steps"]), num_cell(r["peak"]),
            td(str(r["compacted"]), sort=str(r["compacted"])),
            td(str(r["n"]), sort=str(r["n"])),
            td(esc(notes.for_cell(r["agent"], r["model"])), cls="note"),
        ]
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines += ["</tbody>", "</table>"]
    return "\n".join(lines)


def render_ranking(problem, block, notes):
    metric = block["metric"] or "metric"
    unit = block["unit"] or ""
    higher = block["higher_better"]
    unit_h = f" [{unit}]" if unit else ""
    direction = "higher is better" if higher else "lower is better"

    def sort_key(row):
        m = row["metric_mean"]
        if m is None:
            return (1, 0.0, 0.0)
        return (0, -m if higher else m, -(row["pass@1"] or 0))

    rows = sorted(block["rows"], key=sort_key)
    head = ["Agent", "Model", "pass@1", f"{metric}{unit_h} (mean ± sd)", "best", "n", "Notes"]
    numeric = [False, False, True, True, True, True, False]
    lines = [
        f'<h3>{esc(problem)} — ranking metric: <code>{esc(metric)}</code>{esc(unit_h)} '
        f'({direction})</h3>',
        '<table class="sortable">', "<thead><tr>",
    ]
    lines += [th(h, n) for h, n in zip(head, numeric)]
    lines += ["</tr></thead>", "<tbody>"]
    for r in rows:
        best = r["metric_max"] if higher else r["metric_min"]
        mean = r["metric_mean"]
        cells = [
            td(esc(r["agent"])), td(esc(r["model"])),
            frac_cell(r["passed"], r["runs"]),
            td(esc(_fmt_pm(mean, r["metric_std"])),
               sort=f"{mean if mean is not None else (-1e18 if higher else 1e18):.6g}"),
            td(esc(_fmt(best)), sort=f"{best if best is not None else 0:.6g}"),
            td(str(r["metric_n"]), sort=str(r["metric_n"])),
            td(esc(notes.for_cell(r["agent"], r["model"], problem)), cls="note"),
        ]
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines += ["</tbody>", "</table>"]
    return "\n".join(lines)


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent Evaluation Report — {date}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    color: #2a1f1c; background: #fdf7f5; margin: 0; padding: 2rem; line-height: 1.4; }}
  h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; font-size: 1.2rem; }}
  h3 {{ margin-top: 1.4rem; font-size: 1rem; color: #7a6560; }}
  .meta {{ color: #7a6560; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0 1rem;
    background: #fff; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #f0ddd8; padding: 0.35rem 0.55rem; text-align: center; }}
  th:nth-child(1), th:nth-child(2), td:nth-child(1), td:nth-child(2) {{ text-align: left; }}
  td.note {{ text-align: left; color: #7a6560; font-size: 0.8rem; }}
  thead th {{ background: #fde8e6; cursor: pointer; user-select: none;
    position: sticky; top: 0; white-space: nowrap; }}
  thead th:hover {{ background: #f9cbc6; }}
  thead th::after {{ content: "\\2195"; opacity: 0.25; font-size: 0.7em; margin-left: 0.3em; }}
  thead th.asc::after {{ content: "\\2191"; opacity: 0.9; }}
  thead th.desc::after {{ content: "\\2193"; opacity: 0.9; }}
  tbody tr:nth-child(even) {{ background: #fdf7f5; }}
  td.pass {{ background: #edf7f1; color: #2e7d52; font-weight: 600; }}
  td.fail {{ background: #fdf0f0; color: #9b2a2a; }}
  td.partial {{ background: #fff8e8; color: #8a6000; }}
  code {{ background: #fdf3e8; padding: 0 0.2em; border-radius: 3px; }}
</style></head>
<body>
<h1>HPC Coding Agent Benchmark</h1>
<div class="meta">Data through {date} · {n} runs · GPU: {gpu} · deterministic — regenerated from
<code>results/runs.jsonl</code> + <code>results/notes.yaml</code></div>

<h2>pass@1 matrix — by agent × model</h2>
{matrix}

<h2>Per-problem ranking tables</h2>
<p class="meta"><code>n</code> = rounds that produced a measurement (built and ran, passing or
failing). The metric is problem-local — it ranks models within one table only; pass@1 is the
cross-problem axis.</p>
{rankings}

<h2>Token usage — by agent × model</h2>
<p class="meta">Mean per round over the same filtered runs. <code>avg tokens</code> = total
input+output. Cells showing <code>0</code> are gateways whose <code>usage</code> is stripped
(Anthropic shim) — see the note, not a real zero. <code>n</code> = rounds with a stats.json.</p>
{tokens}

<script>
function sortTable(table, col) {{
  var tbody = table.tBodies[0];
  var rows = Array.prototype.slice.call(tbody.rows);
  var numeric = table.tHead.rows[0].cells[col].getAttribute('data-numeric') === '1';
  var cur = table.getAttribute('data-col');
  var dir = (cur == col && table.getAttribute('data-dir') == 'asc') ? 'desc' : 'asc';
  function val(tr) {{
    var c = tr.cells[col];
    var s = c.getAttribute('data-sort');
    return s !== null ? s : c.innerText;
  }}
  rows.sort(function (a, b) {{
    var x = val(a), y = val(b);
    if (numeric) {{ x = parseFloat(x); y = parseFloat(y);
      if (isNaN(x)) x = -Infinity; if (isNaN(y)) y = -Infinity; }}
    if (x < y) return dir == 'asc' ? -1 : 1;
    if (x > y) return dir == 'asc' ? 1 : -1;
    return 0;
  }});
  rows.forEach(function (r) {{ tbody.appendChild(r); }});
  table.setAttribute('data-col', col); table.setAttribute('data-dir', dir);
  var hs = table.tHead.rows[0].cells;
  for (var i = 0; i < hs.length; i++) {{ hs[i].classList.remove('asc', 'desc'); }}
  hs[col].classList.add(dir);
}}
document.querySelectorAll('table.sortable').forEach(function (table) {{
  var hs = table.tHead.rows[0].cells;
  for (var i = 0; i < hs.length; i++) {{
    (function (idx) {{ hs[idx].addEventListener('click', function () {{ sortTable(table, idx); }}); }})(i);
  }}
}});
</script>
</body></html>
"""


def main():
    args = parse_args()
    runs = load_runs(args)
    notes = Notes(args.notes)

    problems = [p for p in PROBLEM_ORDER if any(r["problem"] == p for r in runs)]
    problems += sorted({r["problem"] for r in runs} - set(problems))

    matrix_html = render_matrix(matrix(runs, problems), problems, notes)

    blocks = per_problem_rankings(runs)
    ordered = [p for p in PROBLEM_ORDER if p in blocks]
    ordered += [p for p in sorted(blocks) if p not in PROBLEM_ORDER]
    rankings_html = "\n".join(render_ranking(p, blocks[p], notes) for p in ordered)

    tokens_html = render_tokens(token_rows(runs), notes)

    page = PAGE.format(
        date=report_date(runs), n=len(runs), gpu=esc(args.gpu or "all"),
        matrix=matrix_html, rankings=rankings_html, tokens=tokens_html,
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"wrote {args.out} ({len(runs)} runs, {len(problems)} problems)")


if __name__ == "__main__":
    main()
