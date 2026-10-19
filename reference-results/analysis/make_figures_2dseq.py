#!/usr/bin/env python3
"""Regenerate the 2D-seq figures in results/figures/ (Part 2 of the deck).

    python reference-results/analysis/make_figures_2dseq.py

2D-seq ran on cpu-node — a CPU-only node, no GPU at all — so it is a separate
study from the A40 GPU sweep make_figures.py covers, not folded into it (see
that module's docstring). Scope: the 15 graded rounds across the three
5-round campaigns run 2026-07-27/28 (one 1-round smoke attempt that predates
the 5-round claude+DeepSeek sweep is excluded, same "campaign >= 5 rounds"
rule the main deck uses — see traces/2D-seq/claude/deepseek-ai_DeepSeek-V4-Flash/20260727-200709).

Two figures (pinstride, cost) pull numbers straight out of agent.stream.jsonl
tool-result text rather than a structured .eval/*.json stage — there is no
harness stage for "which core did the OS pin this thread to", so those
numbers are hardcoded constants below with a source path + line comment next
to each one. Everything else comes from results/runs.jsonl exactly like the
main deck.

Needs matplotlib + numpy (same venv as make_figures.py):
    uv venv /tmp/slidesenv --python 3.12
    uv pip install --python /tmp/slidesenv/bin/python matplotlib numpy
    /tmp/slidesenv/bin/python reference-results/analysis/make_figures_2dseq.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (  # noqa: E402  (reuse the main deck's style/save helpers)
    INK, BLUE, GREEN, RED, AMBER, GREY, save, barlabels,
)
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

# --------------------------------------------------------------------------- #
# data: the three graded 5-round campaigns, straight from results/runs.jsonl
# --------------------------------------------------------------------------- #
CAMPAIGNS = {
    ("claude", "DeepSeek-V4-Flash"): "traces/2D-seq/claude/deepseek-ai_DeepSeek-V4-Flash/20260727-212325",
    ("swe-mini", "DeepSeek-V4-Flash"): "traces/2D-seq/swe-mini/deepseek-ai_DeepSeek-V4-Flash/20260728-021132",
    ("claude-pro", "claude-opus-5"): "traces/2D-seq/claude-pro/claude-opus-5/20260728-093535",
}
PAIR_LABEL = {
    ("claude", "DeepSeek-V4-Flash"): "claude + DeepSeek-V4-Flash",
    ("swe-mini", "DeepSeek-V4-Flash"): "swe-mini + DeepSeek-V4-Flash",
    ("claude-pro", "claude-opus-5"): "claude-pro + Opus-5",
}
BASELINE_S = 97.2  # unmodified sequential solver; measured at 97.223591s and 97.288180s
                    # by two different rounds before any edits — see slide 29.


def load_2dseq():
    rows = [json.loads(l) for l in open(os.environ.get("EVAL_RUNS_JSONL", os.path.join(REPO, "results/runs.jsonl")))]
    rows = [r for r in rows if r["problem"] == "2D-seq"]
    rows = [r for r in rows if r["campaign"] in CAMPAIGNS.values()]
    return rows


def stage(r, name):
    for s in r.get("stages", []):
        if s["name"] == name:
            return s
    return None


def solve_time(r):
    s = stage(r, "metric")
    if s and isinstance(s.get("value"), dict):
        return s["value"].get("value")
    return None


ROWS = load_2dseq()


# --------------------------------------------------------------------------- #
# 01 — pass/fail per pairing
# --------------------------------------------------------------------------- #
def fig_pass_status():
    keys = list(CAMPAIGNS)
    cnt = {k: [0, 0] for k in keys}
    for r in ROWS:
        k = (r["agent"], r["model"])
        cnt[k][1] += 1
        cnt[k][0] += r["passed"]
    keys = sorted(keys, key=lambda k: cnt[k][0] / cnt[k][1])
    vals = [100 * cnt[k][0] / cnt[k][1] for k in keys]
    labs = [PAIR_LABEL[k] for k in keys]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    bars = ax.barh(labs, vals, color=[RED if v == 0 else (AMBER if v < 70 else GREEN) for v in vals],
                    height=0.55)
    for b, k in zip(bars, keys):
        ax.text(b.get_width() + 2, b.get_y() + b.get_height() / 2,
                f"{cnt[k][0]}/{cnt[k][1]}  ({b.get_width():.0f}%)", va="center", fontsize=11, color=INK)
    ax.set_xlim(0, 110)
    ax.set_xlabel("pass@1 (%) — 5 graded rounds per pairing")
    ax.set_title("2D-seq on cpu-node (CPU only, 288 cores): who came back\nwith a numerically correct, faster solver")
    save(fig, "2dseq_01_pass_status.png")


# --------------------------------------------------------------------------- #
# 02 — where a 2D-seq round dies
# --------------------------------------------------------------------------- #
def fig_failure_modes():
    # Categorised by hand from each failing round's agent.stream.jsonl / agent.traj.json
    # result event (see traces/2D-seq/**/round-*/agent.stream.jsonl or agent.traj.json).
    cats = [
        ("context overflow\n(“Prompt is too long”, no compaction)", 6, RED),
        ("hit the turn cap\n(error_max_turns)", 1, AMBER),
        ("wrote code that\ndid not compile", 1, BLUE),
    ]
    labels = [c[0] for c in cats]
    vals = [c[1] for c in cats]
    colors = [c[2] for c in cats]
    total_fail = sum(vals)

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=colors, height=0.55)
    for yy, v in zip(y, vals):
        ax.text(v + 0.08, yy, f"{v} / {total_fail} failing rounds", va="center", fontsize=11, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(vals) * 1.55)
    ax.set_xlabel("failing rounds (of 8 total, 15 graded)")
    ax.set_title("75% of 2D-seq failures are the same infra cause —\nand it is not the one the main deck found (slide 6)")
    save(fig, "2dseq_02_failure_modes.png")


# --------------------------------------------------------------------------- #
# 03 — DeepSeek round-by-round solve time
# --------------------------------------------------------------------------- #
def fig_deepseek_progression():
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    pairs = [("claude", "DeepSeek-V4-Flash", BLUE), ("swe-mini", "DeepSeek-V4-Flash", GREEN)]
    w = 0.35
    x = np.arange(1, 6)
    for i, (agent, model, color) in enumerate(pairs):
        rows = {r["round"]: r for r in ROWS if r["agent"] == agent and r["model"] == model}
        heights, fails = [], []
        for rd in x:
            r = rows.get(int(rd))
            t = solve_time(r) if r else None
            if t is None:
                heights.append(BASELINE_S * 1.3)  # nominal bar so the FAILED label has somewhere to sit
                fails.append(True)
            else:
                heights.append(t)
                fails.append(False)
        xs = x + (i - 0.5) * w
        bars = ax.bar(xs, heights, width=w, color=color, alpha=0.9,
                       label=f"{agent} + DeepSeek-V4-Flash",
                       hatch=None)
        for b, f, h in zip(bars, fails, heights):
            if f:
                b.set_color(GREY)
                b.set_hatch("//")
                ax.text(b.get_x() + b.get_width() / 2, h * 1.05, "FAILED",
                        ha="center", va="bottom", fontsize=9, color=INK, rotation=90)
            else:
                ax.text(b.get_x() + b.get_width() / 2, h * 1.08, f"{h:.1f}s",
                        ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(BASELINE_S, color=RED, ls="--", lw=1.4)
    ax.text(0.6, BASELINE_S * 1.05, f"unmodified baseline: {BASELINE_S:.0f} s", color=RED, fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(1, 200)
    ax.set_xticks(x, [f"round {i}" for i in x])
    ax.set_ylabel("SOLVE_TIME (s, log scale)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("DeepSeek-V4-Flash across both harnesses: fast when it finishes,\nswe-mini round 2 shows the risk of dropping -fopenmp unnoticed")
    save(fig, "2dseq_03_deepseek_progression.png")


# --------------------------------------------------------------------------- #
# 04 — the hardware lever: L3 cache residency
# --------------------------------------------------------------------------- #
# Opus-5's own controlled experiment inside its sandbox, both at 144 threads —
# only the pin stride changes (thread t -> core t*stride). Extracted from:
#   traces/2D-seq/claude-pro/claude-opus-5/20260728-093535/round-4/agent.stream.jsonl
#   tool_result text: "144 threads, pinstride=1 (12 L3 slices): SOLVE_TIME 0.821417"
#                      "144 threads, pinstride=2 (24 L3 slices): SOLVE_TIME 0.244308"
# Node facts (node probe): 2x EPYC 9825, 288 cores,
# SMT off, L3 768 MiB across 24 x 32 MiB slices, 12 cores/slice.
def fig_pinstride():
    labels = ["pinstride=1\n144 threads packed onto\n12 of 24 L3 slices", "pinstride=2\n144 threads spread across\nall 24 L3 slices"]
    vals = [0.821417, 0.244308]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    bars = ax.bar(labels, vals, color=[RED, GREEN], width=0.5)
    barlabels(ax, bars, fmt="{:.3f}s")
    ax.set_ylabel("SOLVE_TIME (s)\nsame 144 threads, same binary")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Same core count, same code — {vals[0]/vals[1]:.1f}x from which L3\nslices the threads happened to be pinned onto")
    fig.tight_layout()
    save(fig, "2dseq_04_pinstride.png")


# --------------------------------------------------------------------------- #
# 05 — scored vs. reached
# --------------------------------------------------------------------------- #
def fig_scored_vs_reached():
    labels = ["unmodified\nbaseline", "DeepSeek-V4-Flash\nharness-scored best\n(claude, round 4)",
              "Opus-5 self-timed best\n(round 5 — never replayed,\nsession died first)"]
    vals = [97.2, 3.000696, 0.127420]
    colors = [GREY, GREEN, AMBER]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3g}s", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", xytext=(0, 4), textcoords="offset points",
                    fontsize=11, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.08, 200)
    ax.set_ylabel("SOLVE_TIME (s, log scale)")
    ax.set_title(f"Opus-5 reached {vals[1]/vals[2]:.0f}x faster than the scored result —\n"
                 "and scored zero, because the harness never got to check it")
    hatch_bar = bars[2]
    hatch_bar.set_hatch("//")
    save(fig, "2dseq_05_scored_vs_reached.png")


# --------------------------------------------------------------------------- #
# 06 — cost per round: two harnesses with real cost telemetry
# --------------------------------------------------------------------------- #
# $ cost and cache_read_input_tokens read from each round's final "result"
# event in agent.stream.jsonl (total_cost_usd, usage.cache_read_input_tokens).
# swe-mini's endpoint reports instance_cost=0.0 for every round (the same
# telemetry gap the main deck flags on slide 24) so it is left off this chart.
DS_COST = [43.24, 38.36, 46.72, 32.97, 59.26]     # claude + DeepSeek-V4-Flash, rounds 1-5
OPUS_COST = [4.45, 5.30, 4.16, 6.03, 5.22]        # claude-pro + Opus-5, rounds 1-5


def fig_cost_per_round():
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    data = [DS_COST, OPUS_COST]
    bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], [RED, GREEN]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color(INK)
        med.set_linewidth(2)
    for i, v in enumerate(data, start=1):
        ax.scatter(np.random.normal(i, 0.04, len(v)), v, s=26, color=INK, zorder=3)
    mean_ds, mean_op = np.mean(DS_COST), np.mean(OPUS_COST)
    ax.set_xticks([1, 2], [f"claude + DeepSeek-V4-Flash\n(mean ${mean_ds:.0f}/round)",
                           f"claude-pro + Opus-5\n(mean ${mean_op:.0f}/round)"])
    ax.set_ylabel("$ per round (gateway/subscription billing)")
    ax.set_title(f"DeepSeek costs {mean_ds/mean_op:.1f}x Opus-5 per round for the same task —\nzero cache_read_input_tokens on every DeepSeek round is why")
    fig.text(0.5, -0.02, "DeepSeek re-sends its full conversation as fresh input every turn (no caching); "
                         "Opus-5 reads most of it back from cache.", ha="center", fontsize=9.5, color=GREY)
    fig.tight_layout()
    save(fig, "2dseq_06_cost_per_round.png")


if __name__ == "__main__":
    fig_pass_status()
    fig_failure_modes()
    fig_deepseek_progression()
    fig_pinstride()
    fig_scored_vs_reached()
    fig_cost_per_round()
