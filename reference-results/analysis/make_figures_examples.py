#!/usr/bin/env python3
"""Example/prototype figures for cross-problem performance comparison.

    /tmp/slidesenv/bin/python reference-results/analysis/make_figures_examples.py

Three candidate chart types, all from results/runs.jsonl under the same A40
scope as make_figures.py:

  ex1_gemm_pareto_perf_vs_tokens.png   perf-vs-cost Pareto (tiled-gemm)
  ex2_gemm_best_of_n.png               best-of-n sampling curves (tiled-gemm)
  ex3_gemm_perf_vs_error.png           correctness-vs-performance (tiled-gemm)

Only tiled-gemm carries a real performance metric (cublas_frac);
RabbitCT and stream-compact stages record correctness MSE only,
so they have no perf axis to plot until the harness measures one.
"""

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (  # noqa: E402  (reuse the main deck's style/data)
    INK, BLUE, GREEN, RED, AMBER, GREY, PURPLE, MODEL_SHORT, ROWS_LATEST as ROWS, save, stage,
)
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root


def pair_label(agent, model):
    return f"{agent} + {MODEL_SHORT.get(model, model).splitlines()[0]}"


def gemm_rounds():
    """(agent, model) -> list of (cublas_frac, total_tokens, bench_mse, round)."""
    g = collections.defaultdict(list)
    for r in ROWS:
        if r["problem"] != "tiled-gemm":
            continue
        s = stage(r, "metric")
        if not (s and isinstance(s.get("value"), dict)
                and isinstance(s["value"].get("value"), (int, float))):
            continue
        m = stage(r, "mse")
        mse = m.get("value") if m and isinstance(m.get("value"), (int, float)) else None
        g[(r["agent"], r["model"])].append(
            (s["value"]["value"], r.get("total_tokens") or 0, mse, r["round"]))
    return g


def pareto_upper_left(pts):
    """Frontier for maximize-y / minimize-x: points no other point dominates."""
    front = []
    best_y = -np.inf
    for x, y in sorted(pts):
        if y > best_y:
            front.append((x, y))
            best_y = y
    return front


# --------------------------------------------------------------------------- #
# ex1 — tiled-gemm: best perf vs token cost, Pareto frontier
# --------------------------------------------------------------------------- #
def fig_gemm_pareto():
    g = gemm_rounds()
    pts = {}
    for k, v in g.items():
        toks = [t for _, t, _, _ in v if t > 0]
        if not toks:
            continue
        pts[k] = (float(np.median(toks)), max(f for f, _, _, _ in v))

    front = pareto_upper_left(pts.values())
    fx, fy = zip(*front)

    # manual label nudges (dx pts, dy pts, ha) where the default collides
    nudge = {
        ("pi", "DeepSeek-V4-Flash"): (0, -18, "right"),
        ("swe-mini", "Qwen3.6-27B"): (-10, -4, "right"),
        ("swe-mini", "DeepSeek-V4-Flash"): (0, 8, "right"),
        ("swe-mini", "Llama-3.3-70B-Instruct"): (0, 9, "center"),
        ("claude", "Llama-3.3-70B-Instruct"): (0, -16, "center"),
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.step(fx, fy, where="post", color=GREEN, lw=1.8, zorder=2,
            label="Pareto frontier (cheaper or faster)")
    for k, (x, y) in pts.items():
        on_front = (x, y) in front
        dx, dy, ha = nudge.get(k, (0, 7, "center"))
        ax.scatter(x, y, s=70 if on_front else 42,
                   color=GREEN if on_front else BLUE, alpha=0.9, zorder=3)
        ax.annotate(pair_label(*k), (x, y), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, fontsize=8.5,
                    color=INK if on_front else GREY)
    ax.axhline(0.126, color=RED, ls="--", lw=1.2)
    ax.text(ax.get_xlim()[0], 0.132, "  starter kernel", color=RED, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("median tokens per round (log) — claude/gateway pairs omitted "
                  "(no token accounting)")
    ax.set_ylabel("best achieved fraction of cuBLAS")
    ax.set_ylim(0.05, 1.32)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.set_title("tiled-gemm: performance vs token cost —\n"
                 "the frontier is not the most expensive runs")
    save(fig, "ex1_gemm_pareto_perf_vs_tokens.png")


# --------------------------------------------------------------------------- #
# ex2 — tiled-gemm: best-of-n over independent rounds
# --------------------------------------------------------------------------- #
HILITE = {
    ("claude-pro", "claude-opus-4-8"): RED,
    ("swe-mini", "Kimi-K2.7-Code"): GREEN,
    ("claude", "Qwen3.6-27B"): PURPLE,
}


def fig_gemm_best_of_n():
    g = gemm_rounds()
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for k, v in sorted(g.items()):
        fracs = [f for f, _, _, _ in sorted(v, key=lambda t: t[3])]
        best = np.maximum.accumulate(fracs)
        n = np.arange(1, len(best) + 1)
        c = HILITE.get(k)
        ax.plot(n, best, color=c or GREY, lw=2.2 if c else 1.0,
                alpha=1.0 if c else 0.45, zorder=3 if c else 2,
                label=pair_label(*k) if c else None)
    ax.axhline(1.0, color=INK, ls=":", lw=1.2)
    ax.text(1.0, 1.01, "cuBLAS parity", color=INK, fontsize=9)
    ax.set_xlabel("rounds sampled (independent runs, in campaign order)")
    ax.set_ylabel("best fraction of cuBLAS so far")
    ax.set_xticks(range(1, max(len(v) for v in g.values()) + 1))
    ax.legend(frameon=False, fontsize=10, loc="lower right",
              title="grey: all other pairs")
    ax.set_title("Best-of-n: how much does re-running buy?\n"
                 "swe-mini + Kimi keeps improving; Opus 4.8 is flat after round 2")
    save(fig, "ex2_gemm_best_of_n.png")


# --------------------------------------------------------------------------- #
# ex3 — tiled-gemm: correctness vs performance
# --------------------------------------------------------------------------- #
def fig_gemm_perf_vs_error():
    g = gemm_rounds()
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    floor = 1e-14
    worst = None
    for k, v in g.items():
        for f, _, mse, _ in v:
            if mse is None:
                continue
            bad = mse > 0.01
            ax.scatter(f, max(mse, floor), s=34,
                       color=RED if bad else BLUE, alpha=0.6, zorder=3)
            if bad and (worst is None or f > worst[0]):
                worst = (f, mse, pair_label(*k))
    if worst:
        ax.annotate(f"{worst[2]}: {worst[0]:.0%} of cuBLAS,\nbut MSE {worst[1]:.0f}",
                    worst[:2], xytext=(-15, -35), textcoords="offset points",
                    ha="right", fontsize=9, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.axhline(0.01, color=RED, ls="--", lw=1.4)
    ax.text(0.02, 0.014, "pass threshold: MSE ≤ 0.01", color=RED, fontsize=10)
    ax.axvline(0.126, color=GREY, ls=":", lw=1.2)
    ax.text(0.13, floor * 2, "starter kernel", color=GREY, fontsize=9, rotation=90)
    ax.set_yscale("log")
    ax.set_xlabel("achieved fraction of cuBLAS")
    ax.set_ylabel(f"benchmark MSE (log; 0 plotted at {floor:g})")
    ax.set_title("Why perf alone would mis-rank: two near-cuBLAS kernels\n"
                 "are wrong — the MSE gate catches them")
    save(fig, "ex3_gemm_perf_vs_error.png")


if __name__ == "__main__":
    fig_gemm_pareto()
    fig_gemm_best_of_n()
    fig_gemm_perf_vs_error()
