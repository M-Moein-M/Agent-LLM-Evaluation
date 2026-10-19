#!/usr/bin/env python3
"""Emit the per-problem ranking tables (markdown) for report.md.

One table per problem: agent × model, with the binary pass@1 next to the
continuous ranking measurement (mean ± sample stdev over every round that
produced a number — passing AND failing; see per_problem_rankings).

    python reference-results/analysis/collect_results.py --traces traces    # writes results/runs.jsonl
    python reference-results/analysis/report_tables.py > results/tables.md

The ranking metric is whatever the problem's ``check.py`` declared via
``metric_stage`` — this script needs no per-problem knowledge, and each problem
is free to pick its own. That metric is INTRA-problem: it orders models within
one table only. pass@1 is the axis that compares across problems.

Defaults to the A40, the only device in scope.
"""

import argparse
import json
from collections import Counter

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))  # collect_results
from collect_results import CALL_LIMIT, per_problem_rankings

# Table order = the order the problems appear in report.md.
PROBLEM_ORDER = ["RabbitCT-LolaBunny", "RabbitCT-LolaOMP", "stream-compact", "tiled-gemm"]

# Campaigns dropped as harness/infra artifacts, not capability signal. The
# claude-pro subscription hit "You've hit your session limit" mid-sweep on these
# two 5-round campaigns (so they survive --min-campaign-rounds but contain a
# truncated build-fail round with no solution.json); each was re-run clean and
# is superseded by the campaign named alongside it. Excluding them keeps
# claude-pro at the reported 5 rounds/problem (/20). See report.md §8.
EXCLUDE_CAMPAIGNS = {
    # LolaOMP: 012633 r5 = session-limit build-fail; superseded by 20260716-181530.
    "RabbitCT-LolaOMP/claude-pro/claude-opus-4-8/20260716-012633",
    # tiled-gemm: 040522 r5 = session-limit build-fail; superseded by 20260716-122816.
    "tiled-gemm/claude-pro/claude-opus-4-8/20260716-040522",
    # pi+DeepSeek-V4-Flash-0731 20260815-165955 = 0/20, every round build-fails
    # (bad checkpoint/endpoint state at the time); superseded by the earlier
    # clean 20260812-005922 run (17/20). Pin to that one, not "latest".
    "RabbitCT-LolaBunny/pi/deepseek-ai_DeepSeek-V4-Flash-0731/20260815-165955",
    "RabbitCT-LolaOMP/pi/deepseek-ai_DeepSeek-V4-Flash-0731/20260815-165955",
    "stream-compact/pi/deepseek-ai_DeepSeek-V4-Flash-0731/20260815-165955",
    "tiled-gemm/pi/deepseek-ai_DeepSeek-V4-Flash-0731/20260815-165955",
    # kimi+gemma-4-31B-it-FP8-block 20260815-154047 = 0/20, every round hits
    # "the provider stream was interrupted" mid tool-call (endpoint cut off
    # before the final Write completed, leaving truncated/missing source ->
    # build fail); superseded by the earlier clean 20260801-100814 run (7/20).
    # Pin to that one, not "latest".
    "RabbitCT-LolaBunny/kimi/gemmaselfhosted_google_gemma-4-31B-it/20260815-154047",
    "RabbitCT-LolaOMP/kimi/gemmaselfhosted_google_gemma-4-31B-it/20260815-154047",
    "stream-compact/kimi/gemmaselfhosted_google_gemma-4-31B-it/20260815-154047",
    "tiled-gemm/kimi/gemmaselfhosted_google_gemma-4-31B-it/20260815-154047",
}


def _campaign_key(run):
    """Campaign path relative to the trace root: <problem>/<agent>/<model>/<date>.

    ``run["campaign"]`` is an absolute-ish path ending in that 4-segment tail; we
    match on the tail so the exclusion set is independent of where traces live.
    """
    camp = (run.get("campaign") or "").rstrip("/")
    parts = camp.split("/")
    return "/".join(parts[-4:]) if len(parts) >= 4 else camp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="results/runs.jsonl", help="Input JSONL from collect_results.py")
    parser.add_argument("--gpu", default="A40", help="Only include runs on this GPU ('' = all)")
    parser.add_argument(
        "--min-campaign-rounds",
        type=int,
        default=5,
        help="Drop campaigns with fewer rounds than this. A scored sweep is 10 "
             "rounds; the short ones are harness bring-up/smoke runs (often a "
             "single round) and would otherwise appear as real rows",
    )
    return parser.parse_args()


def _fmt(value, digits=3):
    if value is None:
        return "—"
    if value == 0:
        return "0"
    return f"{value:.{digits}g}"


def _fmt_pm(mean, std, digits=3):
    """``mean ± std``; a single passing round has no spread, so it shows as n/a."""
    if mean is None:
        return "—"
    if std is None:
        return f"{_fmt(mean, digits)} ± n/a"
    return f"{_fmt(mean, digits)} ± {_fmt(std, digits)}"


def _fmt_limits(limits):
    """Non-zero limit hits of one row, e.g. ``3 capped, 1 >120 calls``; ``—`` if none."""
    parts = []
    if limits.get("capped_calls"):
        parts.append(f"{limits['capped_calls']} capped")
    if limits.get("over_call_limit"):
        parts.append(f"{limits['over_call_limit']} >{CALL_LIMIT} calls")
    if limits.get("hung"):
        parts.append(f"{limits['hung']} hung")
    if limits.get("empty_reply_nudges"):
        parts.append(f"{limits['empty_reply_nudges']} nudges")
    return ", ".join(parts) or "—"


def render(problem, block):
    metric = block["metric"] or "metric"
    unit = block["unit"] or ""
    higher_better = block["higher_better"]
    direction = "higher is better" if higher_better else "lower is better"

    # Rank by the metric; models that never got far enough to be measured sink to
    # the bottom (nothing to rank), ordered by pass@1 among themselves.
    def sort_key(row):
        m = row["metric_mean"]
        if m is None:
            return (1, 0.0, 0.0)
        return (0, -m if higher_better else m, -(row["pass@1"] or 0))

    rows = sorted(block["rows"], key=sort_key)

    header_unit = f" [{unit}]" if unit else ""
    lines = [
        f"#### {problem} — ranking metric: `{metric}`{header_unit} ({direction})",
        "",
        f"| Agent | Model | pass@1 | {metric}{header_unit} (mean ± sd) | best | n | limits |",
        "|---|---|:--:|:--:|:--:|:--:|:--|",
    ]
    for r in rows:
        best = r["metric_max"] if higher_better else r["metric_min"]
        lines.append(
            f"| {r['agent']} | {r['model']} | "
            f"{r['passed']}/{r['runs']} | "
            f"{_fmt_pm(r['metric_mean'], r['metric_std'])} | "
            f"{_fmt(best)} | {r['metric_n']} | {_fmt_limits(r['limits'])} |"
        )
    lines.append("")
    lines.append(
        "> `n` = rounds that produced a measurement — every round that built and "
        "ran, **passing or failing**. Rounds that failed at build/run never "
        "measured anything and are in pass@1's denominator only, so `n < runs` "
        "there. The metric is problem-local: it ranks models *within* this table "
        "and is not comparable across problems — pass@1 is the cross-problem axis."
    )
    lines.append("")
    lines.append(
        f"> `limits` = model calls stopped by the output cap (`capped`, summed "
        f"over rounds; unknown for claude and old kimi rounds), rounds over "
        f"{CALL_LIMIT} model calls, rounds where kimi hung after its final reply "
        f"(`hung`, pass verdict kept), and claude's re-prompts after a "
        f"thinking-only reply (`nudges`, not counted by its turn limit)."
    )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    with open(args.runs, "r", encoding="utf-8") as fh:
        runs = [json.loads(line) for line in fh if line.strip()]

    if args.gpu:
        runs = [r for r in runs if args.gpu in r.get("gpu", "")]

    runs = [r for r in runs if _campaign_key(r) not in EXCLUDE_CAMPAIGNS]

    sizes = Counter(r.get("campaign") for r in runs)
    runs = [r for r in runs if sizes[r.get("campaign")] >= args.min_campaign_rounds]

    blocks = per_problem_rankings(runs)
    ordered = [p for p in PROBLEM_ORDER if p in blocks]
    ordered += [p for p in sorted(blocks) if p not in PROBLEM_ORDER]

    for problem in ordered:
        print(render(problem, blocks[problem]))


if __name__ == "__main__":
    main()
