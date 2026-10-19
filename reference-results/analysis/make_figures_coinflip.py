#!/usr/bin/env python3
"""Intro-slide figure: a coin-flip confidence interval, same look as 24b.

    /tmp/slidesenv/bin/python reference-results/analysis/make_figures_coinflip.py

Standalone on purpose. Reuses the deck style and the Wilson helper so the
dot-and-whisker matches 24a/24b, but touches nothing else.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_figures import BLUE, INK, save  # noqa: E402  (style + rcParams load here)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig_coinflip_ci():
    k, n = 7, 12
    pt = 100 * k / n
    lo, hi = (100 * x for x in wilson(k, n))

    fig, ax = plt.subplots(figsize=(10.8, 2.6))
    y = 0

    # candidate "true rates" the audience might guess
    for x, lab, ty, ha in [(50, "fair coin 50%", 0.74, "right"),
                           (58, "sample 58%", 0.50, "center"),
                           (70, "70%", 0.74, "left")]:
        ax.axvline(x, color=INK, linewidth=1.0, linestyle="--", alpha=0.35, zorder=1)
        ax.text(x, ty, lab, ha=ha, va="bottom", fontsize=9, color=INK)

    # the interval
    ax.hlines(y, lo, hi, color=BLUE, linewidth=2.4, alpha=0.55, zorder=2)
    ax.scatter([pt], [y], s=70, color=BLUE, zorder=3)
    ax.vlines([lo, hi], y - 0.12, y + 0.12, color=BLUE, linewidth=1.8, zorder=3)
    ax.text(hi + 2, y,
            f"{pt:.0f}% observed   95% interval [{lo:.0f}–{hi:.0f}]   7 of 12 flips",
            va="center", fontsize=10, color=INK)

    ax.set_yticks([])
    ax.set_ylim(-0.45, 1.15)
    ax.set_xlim(0, 100)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("true heads rate (%), 95% confidence interval")
    ax.set_title("Flip a coin 12 times, get 7 heads — what is the true rate?")

    fig.text(0.01, -0.06,
             "A small sample gives an estimate, not the true rate. 7 of 12 points at "
             "58%, but 50%, 58% and 70% all sit inside the 95% interval, so the flips "
             "cannot tell them apart. The bar is the range of true rates the sample is\n"
             "consistent with. Every pass rate in this deck is read the same way.",
             fontsize=8, color=INK, va="top", ha="left")
    fig.subplots_adjust(bottom=0.30, top=0.82)
    save(fig, "ex5_coinflip_ci.png")


if __name__ == "__main__":
    fig_coinflip_ci()
