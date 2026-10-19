#!/usr/bin/env python3
"""Statistical companion figures (20-29) for the agent-eval deck.

20c, 21, 22a, 22b, 23b and 25 were archived after review; their code is gone
with them. 22c carries what 22a and 22b showed, in one figure.

    /tmp/slidesenv/bin/python reference-results/analysis/make_figures_stats.py

Separate file on purpose: every figure here is *additive*. Nothing in
make_figures.py is touched, and none of the existing 24 PNGs are rewritten,
so the pass@1 views stay exactly as they are while these answer the questions
they cannot: how big is the uncertainty, what kind of failure is it, does
retrying help, and does the ranking survive the unbalanced design.

Same population as the rest of the deck -- make_figures.load_latest_all()
via ROWS_LATEST (A40 only, latest campaign per pairing, 1,132 rounds / 476
passed). Every failing round stays in the denominator, per slide 16.

No scipy on the cluster, so Wilson intervals, the two-sided Fisher exact test
and the logistic fit are all hand-rolled (stdlib math.comb + numpy).
"""

import collections
import math
import os
import statistics as st
import sys
import textwrap
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import make_figures as M
from make_figures import (AMBER, BLUE, GREEN, GREY, INK, PURPLE, RED,
                          ROWS_LATEST, display_agent, save)

np.random.seed(0)


# --------------------------------------------------------------------------- #
# statistics helpers
# --------------------------------------------------------------------------- #
def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion, as fractions.

    The range of true pass rates consistent with k of n. Preferred over the
    normal approximation because our cells are small (n=20 is typical) and
    several sit at or near 0/1, where the normal interval runs off the end of
    the scale or collapses to zero width."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]].

    Exact conditional test, so it stays valid at the cell sizes we have where
    a chi-square would not. Sums hypergeometric probabilities at least as
    extreme as observed."""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    if n == 0 or r1 == 0 or c1 == 0:
        return 1.0

    def p(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    obs = p(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs + 1e-12))


def pass_at_k(n, c, k):
    """Unbiased pass@k from n attempts of which c passed: 1 - C(n-c,k)/C(n,k).

    The probability that a random sample of k of those n rounds contains at
    least one pass. Standard estimator (Chen et al. 2021)."""
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# --------------------------------------------------------------------------- #
# logistic fit: passed ~ agent + model + problem
# --------------------------------------------------------------------------- #
class Design:
    """Indicator coding of (agent, model, problem) with the first level of
    each factor as reference. Levels are fixed at construction so bootstrap
    resamples keep the same columns even when a level goes missing."""

    def __init__(self, rows):
        self.agents = sorted({display_agent(r) for r in rows})
        self.models = sorted({r["model"] for r in rows})
        self.problems = sorted({r["problem"] for r in rows})
        self.ai = {a: i for i, a in enumerate(self.agents)}
        self.mi = {m: i for i, m in enumerate(self.models)}
        self.pi = {p: i for i, p in enumerate(self.problems)}
        self.ncol = 1 + (len(self.agents) - 1) + (len(self.models) - 1) + (len(self.problems) - 1)

    def index(self, rows):
        a = np.array([self.ai[display_agent(r)] for r in rows])
        m = np.array([self.mi[r["model"]] for r in rows])
        p = np.array([self.pi[r["problem"]] for r in rows])
        y = np.array([1.0 if r["passed"] else 0.0 for r in rows])
        return a, m, p, y

    def matrix(self, a, m, p):
        n = len(a)
        X = np.zeros((n, self.ncol))
        X[:, 0] = 1.0
        oa, om = 1, len(self.agents)
        op = om + len(self.models) - 1
        for k in range(n):
            if a[k] > 0:
                X[k, oa + a[k] - 1] = 1.0
            if m[k] > 0:
                X[k, om + m[k] - 1] = 1.0
            if p[k] > 0:
                X[k, op + p[k] - 1] = 1.0
        return X


def irls(X, y, lam=1.0, iters=300):
    """Ridge-penalised logistic regression by Newton / IRLS.

    lam > 0 is required, not cosmetic: Llama-3.3-70B passes 0 of its rounds,
    which drives its coefficient to -inf under plain maximum likelihood and
    the solve to a singular matrix. The ridge term keeps it finite, at the
    cost of shrinking extreme levels toward the mean -- which is why Llama
    shows a few percent rather than a flat zero."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ b))
        W = np.clip(p * (1 - p), 1e-8, None)
        A = X.T @ (X * W[:, None]) + lam * np.eye(len(b))
        s = np.linalg.solve(A, X.T @ (y - p) - lam * b)
        b += s
        if np.max(np.abs(s)) < 1e-10:
            break
    return b


def adjusted_rates(des, b, a, m, p):
    """G-computation: rewrite every round's agent (or model) to one level,
    keep the other factors as observed, average the predicted pass
    probability. Answers "if this agent had run every model on every problem,
    at the mix we actually ran" -- a percentage on the same scale as the
    pass@1 bars, not a log-odds coefficient."""
    out = {}
    for i, name in enumerate(des.agents):
        X = des.matrix(np.full(len(a), i), m, p)
        out[("agent", name)] = float(np.mean(1 / (1 + np.exp(-X @ b))))
    for i, name in enumerate(des.models):
        X = des.matrix(a, np.full(len(a), i), p)
        out[("model", name)] = float(np.mean(1 / (1 + np.exp(-X @ b))))
    return out


def fit_adjusted(rows, nboot=2000, lam=1.0):
    """Point estimates plus percentile bootstrap CIs (resample rounds with
    replacement, refit each draw). 0.009 s per draw at n=1,132, so ~18 s."""
    des = Design(rows)
    a, m, p, y = des.index(rows)
    b = irls(des.matrix(a, m, p), y, lam)
    point = adjusted_rates(des, b, a, m, p)

    draws = collections.defaultdict(list)
    n = len(rows)
    for _ in range(nboot):
        idx = np.random.randint(0, n, n)
        bb = irls(des.matrix(a[idx], m[idx], p[idx]), y[idx], lam)
        for key, val in adjusted_rates(des, bb, a[idx], m[idx], p[idx]).items():
            draws[key].append(val)
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
          for k, v in draws.items()}
    return des, b, point, ci


def _mlabel(m):
    return M._model_label(m, {"gemma-4-31B-it-FP8-block": "gemma-4-31B-it (FP8)"})


def _pair_label(a, m):
    return f"{a} + {_mlabel(m)}"


def _caption(fig, text, fontsize=8, axes_bottom_in=0.75):
    """Figure footnote, hung just under the figure box.

    Both offsets are in inches, not figure fractions: matplotlib's default
    bottom margin is 11% of the figure, which is a couple of blank inches
    under the tall stacked-bar figures here. savefig already runs with
    bbox_inches="tight", so the text at a small negative y is cropped in."""
    w, h = fig.get_size_inches()
    if axes_bottom_in:
        fig.subplots_adjust(bottom=axes_bottom_in / h)
    # A caption line wider than the axes drags the tight bbox out with it.
    cols = max(60, int(w / (0.0072 * fontsize)))
    wrapped = "\n".join("\n".join(textwrap.wrap(line, cols)) or ""
                        for line in text.split("\n"))
    fig.text(0.005, -0.15 / h, wrapped, fontsize=fontsize, color=INK,
             va="top", ha="left")


def _dotwhisker(ax, labels, point, lo, hi, color=BLUE, note=None):
    """Horizontal dot-and-whisker, best at the top."""
    y = np.arange(len(labels))[::-1]
    ax.hlines(y, lo, hi, color=color, linewidth=2.2, alpha=0.55, zorder=2)
    ax.scatter(point, y, s=54, color=color, zorder=3)
    for yy, lo_, hi_ in zip(y, lo, hi):
        ax.vlines([lo_, hi_], yy - 0.14, yy + 0.14, color=color, linewidth=1.6, zorder=3)
    ax.set_yticks(y, labels, fontsize=10)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    if note:
        for yy, t in zip(y, note):
            ax.text(ax.get_xlim()[1], yy, t, va="center", ha="right", fontsize=9, color=INK)


# --------------------------------------------------------------------------- #
# 20a -- pass@1 by agent, with Wilson intervals
# --------------------------------------------------------------------------- #
def fig_pass1_by_agent_ci():
    # Same pooled counts as 17a (which stays untouched); the bar is replaced
    # by a point and the range of true pass rates the sample is consistent
    # with. Reading: overlapping whiskers mean the visual gap between two
    # bars in 17a is not evidence of a difference.
    cnt = collections.defaultdict(lambda: [0, 0])
    for r in ROWS_LATEST:
        c = cnt[r["agent"]]
        c[1] += 1
        c[0] += r["passed"]
    keys = sorted(cnt, key=lambda k: cnt[k][0] / cnt[k][1], reverse=True)
    pt = np.array([100 * cnt[k][0] / cnt[k][1] for k in keys])
    ci = [wilson(*cnt[k]) for k in keys]
    lo = np.array([100 * c[0] for c in ci])
    hi = np.array([100 * c[1] for c in ci])

    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    _dotwhisker(ax, keys, pt, lo, hi)
    for i, k in enumerate(keys):
        p, n = cnt[k]
        ax.text(hi[i] + 1.2, len(keys) - 1 - i,
                f"{pt[i]:.1f}%   n={n}   [{lo[i]:.1f}, {hi[i]:.1f}]",
                va="center", fontsize=9.5, color=INK)
    ax.set_xlim(20, 82)
    ax.set_xlabel("pass@1 (%) with 95% Wilson interval")
    ax.set_title("pass@1 by agent, with sampling uncertainty")

    # Overlapping whiskers are suggestive, not a test -- run the test.
    tests = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pa, na = cnt[keys[i]]
            pb, nb = cnt[keys[j]]
            pv = fisher_exact(pa, na - pa, pb, nb - pb)
            tests.append(f"{keys[i]} vs {keys[j]}: p={pv:.3f}" + ("*" if pv < 0.05 else ""))
    _caption(fig,
             "Whiskers are the range of true pass rates consistent with the observed count. "
             "Pooled over every model each agent ran, so the mix differs between rows — "
             "24a adjusts for that.\nFisher exact, all pairs (* = p < 0.05):  "
             + "   ".join(tests) +
             "\nOnly swe-mini vs claude separates at this sample size. The other five gaps "
             "are not evidence of a difference.")
    save(fig, "20a_pass1_by_agent_ci.png")


# --------------------------------------------------------------------------- #
# 20b -- pass@1 by model, with Wilson intervals
# --------------------------------------------------------------------------- #
def fig_pass1_by_model_ci():
    # Mirror of 20a on the model axis, same exclusion rule 17b uses
    # (claude-opus-4-8 is a single pairing, not a rollup).
    cnt = collections.defaultdict(lambda: [0, 0])
    for r in ROWS_LATEST:
        if r["model"] == "claude-opus-4-8":
            continue
        c = cnt[r["model"]]
        c[1] += 1
        c[0] += r["passed"]
    keys = sorted(cnt, key=lambda k: cnt[k][0] / cnt[k][1], reverse=True)
    pt = np.array([100 * cnt[k][0] / cnt[k][1] for k in keys])
    ci = [wilson(*cnt[k]) for k in keys]
    lo = np.array([100 * c[0] for c in ci])
    hi = np.array([100 * c[1] for c in ci])

    fig, ax = plt.subplots(figsize=(9.4, 0.52 * len(keys) + 1.8))
    _dotwhisker(ax, [_mlabel(k) for k in keys], pt, lo, hi)
    for i, k in enumerate(keys):
        p, n = cnt[k]
        ax.text(hi[i] + 1.5, len(keys) - 1 - i,
                f"{pt[i]:.0f}%  n={n}  [{lo[i]:.0f}, {hi[i]:.0f}]",
                va="center", fontsize=9, color=INK)
    ax.set_xlim(-3, 118)
    ax.set_xlabel("pass@1 (%) with 95% Wilson interval")
    ax.set_title("pass@1 by model, with sampling uncertainty")
    _caption(fig,
             "Pooled over every agent that ran the model; claude-opus-4-8 dropped because only "
             "one agent ran it (same rule 17b uses).\n"
             "The top five intervals all overlap — they are one tier, not a ranking. The bottom "
             "four separate cleanly from them and from each other.")
    save(fig, "20b_pass1_by_model_ci.png")


# --------------------------------------------------------------------------- #
# pass@k table, feeds 22c
# --------------------------------------------------------------------------- #
def _passk_table(kmax=5):
    """pairing -> {k: pass@k}, averaged over the four problems so a pairing
    that always fails one problem cannot be rescued by extra draws on the
    others. Restricted to pairings that ran all four problems with at least
    kmax rounds each."""
    per = collections.defaultdict(lambda: [0, 0])
    for r in ROWS_LATEST:
        # display_agent, not r["agent"], so the Opus rounds read "claude pro"
        # here as they do on 20a/24a. Pure relabel: the key already carries the
        # model, and every Opus round has model = claude-opus-4-8.
        c = per[(display_agent(r), r["model"], r["problem"])]
        c[1] += 1
        c[0] += r["passed"]
    probs = collections.defaultdict(dict)
    for (a, m, p), (c, n) in per.items():
        probs[(a, m)][p] = (n, c)
    out = {}
    for pair, d in probs.items():
        if len(d) < 4 or min(n for n, _ in d.values()) < kmax:
            continue
        out[pair] = {k: 100 * st.mean(pass_at_k(n, c, k) for n, c in d.values())
                     for k in range(1, kmax + 1)}
    return out


# --------------------------------------------------------------------------- #
# 22c -- retry gain broken out per attempt
# --------------------------------------------------------------------------- #
# The archived 22b showed the k=1 -> k=5 jump as one green block, which hid
# whether the gain arrives on the second attempt or trickles in over five. That
# is the operational question: a pairing that banks everything at k=2 wants one
# retry, not four. Same bars, the block split into its four increments -- so this
# also carries what the archived 22a curve showed, per pairing.
K_COLORS = [GREEN, BLUE, AMBER, PURPLE]


def fig_retry_gain_by_k(kmax=5):
    tab = _passk_table(kmax)
    keys = sorted(tab, key=lambda k: tab[k][kmax] - tab[k][1])
    base = [tab[k][1] for k in keys]
    # inc[j] is the pass@(j+2) - pass@(j+1) column, one bar segment per k step
    inc = [[tab[k][j + 2] - tab[k][j + 1] for k in keys] for j in range(kmax - 1)]

    fig, ax = plt.subplots(figsize=(11.6, 0.36 * len(keys) + 2.6))
    y = np.arange(len(keys))
    ax.barh(y, base, height=0.6, color=GREY, label="pass@1")
    left = np.array(base, dtype=float)
    for j, col in enumerate(inc):
        ax.barh(y, col, left=left, height=0.6, color=K_COLORS[j % len(K_COLORS)],
                label=f"+ k={j + 2}")
        for i, v in enumerate(col):
            # A segment under ~2.5 pp is narrower than its own label.
            if v >= 2.5:
                ax.text(left[i] + v / 2, y[i], f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white")
        left += np.array(col, dtype=float)

    for i, k in enumerate(keys):
        gain = tab[k][kmax] - tab[k][1]
        ax.text(left[i] + 1.2, y[i], f"+{gain:.0f} pp", va="center",
                fontsize=8.5, color=INK)

    # Same flat test the archived 22a used: no real gain while off both rails.
    labels = [_pair_label(*k) + ("  \u2190flat" if tab[k][kmax] - tab[k][1] < 3
                                and 2 < tab[k][kmax] < 95 else "")
              for k in keys]
    ax.set_yticks(y, labels, fontsize=8.5)
    for tick, k in zip(ax.get_yticklabels(), keys):
        if tab[k][kmax] - tab[k][1] < 3 and 2 < tab[k][kmax] < 95:
            tick.set_color(RED)
    ax.set_xlim(0, 118)
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.set_xlabel("pass rate (%)")
    ax.set_title("Where the retry gain arrives, attempt by attempt", pad=26)
    # Above the axes: "lower right" lands on the bottom bar, which is a real bar.
    ax.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.002),
              frameon=False, ncol=5)
    ax.grid(axis="y", visible=False)
    _caption(fig,
             "Each coloured block is pass@k minus pass@(k-1), the pass rate bought by the "
             "k-th attempt alone. Unbiased estimator 1 - C(n-c,k)/C(n,k) per problem, then "
             f"averaged over the 4 problems. Segments under 2.5 pp are drawn but not labelled. "
             "Blocks shrinking left to right is the normal shape: the second attempt catches "
             "most of the flakiness and the rest is diminishing returns, so a retry budget of "
             "1 buys nearly what a budget of 4 does. A pairing whose blocks stay even is still "
             "climbing at k=5 and would keep paying for more attempts. "
             "Red \u2190flat marks under 3 pp of total gain while off both rails: the same "
             "problem fails every round, and no retry policy reaches it.")
    save(fig, "22c_retry_gain_by_k.png")


def fig_retry_gain_by_k_h(kmax=5, top=20):
    """22c turned on its side and cut to the pairings that gained the most.

    The row version is ~15 in tall, which is portrait on a landscape slide, and
    all 37 pairings is more chart than the point needs. Same table and same
    arithmetic, but the bars grow up instead of right, the order is reversed to
    read high gain first, and only the top `top` pairings are drawn.

    The cut takes the flat rows and the four Llama pairings with it, so the red
    flat marking 22c carries is gone here -- 22c stays the figure that shows the
    pairings retries cannot help. `cut` is the gain of the last pairing kept, so
    the caption can state the threshold instead of asserting one.
    """
    tab = _passk_table(kmax)
    keys = sorted(tab, key=lambda k: -(tab[k][kmax] - tab[k][1]))
    dropped = len(keys) - min(top, len(keys))
    cut = tab[keys[top - 1]][kmax] - tab[keys[top - 1]][1] if dropped else 0
    keys = keys[:top]
    base = [tab[k][1] for k in keys]
    inc = [[tab[k][j + 2] - tab[k][j + 1] for k in keys] for j in range(kmax - 1)]

    fig, ax = plt.subplots(figsize=(0.42 * len(keys) + 1.6, 7.2))
    x = np.arange(len(keys))
    ax.bar(x, base, width=0.66, color=GREY, label="pass@1")
    bot = np.array(base, dtype=float)
    for j, col in enumerate(inc):
        ax.bar(x, col, bottom=bot, width=0.66, color=K_COLORS[j % len(K_COLORS)],
               label=f"+ k={j + 2}")
        for i, v in enumerate(col):
            if v >= 2.5:
                ax.text(x[i], bot[i] + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white")
        bot += np.array(col, dtype=float)

    for i, k in enumerate(keys):
        gain = tab[k][kmax] - tab[k][1]
        ax.text(x[i], bot[i] + 1.5, f"+{gain:.0f} pp", ha="center", va="bottom",
                rotation=90, fontsize=7.5, color=INK)

    ax.set_xticks(x, [_pair_label(*k) for k in keys], fontsize=8.5, rotation=40,
                  ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 122)
    ax.set_xlim(-0.7, len(keys) - 0.3)
    ax.set_ylabel("pass rate (%)")
    ax.set_title(f"Where the retry gain arrives, attempt by attempt — "
                 f"top {len(keys)} pairings", pad=26)
    ax.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.002),
              frameon=False, ncol=5)
    ax.grid(axis="x", visible=False)
    _caption(fig,
             "Column version of 22c, same numbers, sorted by total retry gain and cut to the "
             f"top {len(keys)}. Each coloured block is pass@k minus pass@(k-1), the pass rate "
             "bought by the k-th attempt alone. Unbiased estimator 1 - C(n-c,k)/C(n,k) per "
             "problem, then averaged over the 4 problems. Segments under 2.5 pp are drawn but "
             "not labelled.\n"
             "Blocks shrinking bottom to top is the normal shape: the second attempt catches "
             "most of the flakiness and the rest is diminishing returns, so a retry budget of "
             "1 buys nearly what a budget of 4 does. A pairing whose blocks stay even is still "
             "climbing at k=5 and would keep paying for more attempts.\n"
             f"{dropped} of the {dropped + len(keys)} pairings are not drawn, every one at or "
             f"below +{cut:.0f} pp of total gain. That cut removes the four Llama-3.3-70B "
             "pairings, which pass 0 rounds at every k, and the pairings retries cannot move at "
             "all. 22c keeps all of them and marks those in red.",
             axes_bottom_in=1.55)
    save(fig, "22c-h_retry_gain_by_k.png")


# --------------------------------------------------------------------------- #
# 23a -- how failing rounds ended
# --------------------------------------------------------------------------- #
# Figure 03 buckets by *where the harness caught* the failure. That hides the
# distinction the fix depends on: an agent that ran out of room is a budget
# problem, an agent that stopped and declared itself done is a capability
# problem. end_reason is already on every row.
END_SAID_DONE = {"completed"}
END_CAP = {"context_overflow", "max_turns", "max_steps_exceeded", "output_limit",
           "cut_off", "limits_exceeded", "timeout"}
END_PROTO = {"api_error", "thinking_block", "empty_response", "rate_limit",
             "connection_error", "format_error"}
END_SEG = [('agent said "done"', RED), ("hit a cap", AMBER),
           ("provider / protocol error", GREY), ("no end reason recorded", "#d8dde3")]


def _end_seg(r):
    e = r.get("end_reason") or ""
    if e in END_SAID_DONE:
        return 'agent said "done"'
    if e in END_CAP:
        return "hit a cap"
    if e in END_PROTO:
        return "provider / protocol error"
    return "no end reason recorded"


def _fail_cluster(r):
    """Figure 03's cluster, with its biggest bar split in two.

    "Agent never finished" merges two different problems with two different
    fixes: an agent that ran out of room, and one that handed in nothing.
    """
    slug = r.get("fail_class") or "unknown"
    c = M.FAIL_CLUSTER.get(slug, "Infra hiccup")
    if c == "Agent never finished":
        return ("Stopped early, produced nothing"
                if slug in ("no_solution", "incomplete_solution")
                else "Ran out of budget")
    return c


def fig_failure_by_end_reason():
    fails = [r for r in ROWS_LATEST if not r["passed"]]
    total = len(fails)

    grid = collections.defaultdict(collections.Counter)
    for r in fails:
        c = _fail_cluster(r)
        if c == "Infra hiccup":
            continue
        grid[c][_end_seg(r)] += 1
    keys = sorted(grid, key=lambda k: sum(grid[k].values()), reverse=True)
    tot = [sum(grid[k].values()) for k in keys]

    fig, ax = plt.subplots(figsize=(11.0, 0.78 * len(keys) + 2.4))
    y = np.arange(len(keys))[::-1]
    left = np.zeros(len(keys))
    for name, color in END_SEG:
        w = np.array([grid[k][name] for k in keys], dtype=float)
        if w.sum() == 0:
            continue
        ax.barh(y, w, left=left, height=0.55, color=color, label=name,
                edgecolor="white", linewidth=0.5)
        txt = INK if name == "no end reason recorded" else "white"
        for yy, ww, ll in zip(y, w, left):
            if ww >= 12:
                ax.text(ll + ww / 2, yy, f"{int(ww)}", ha="center", va="center",
                        fontsize=9, color=txt, fontweight="bold")
        left += w
    for yy, t, l in zip(y, tot, left):
        ax.text(l + total * 0.012, yy, f"{t}  ({100*t/total:.0f}%)",
                va="center", fontsize=10, color=INK)

    labs = [f"{k}\n{_END_EXAMPLE.get(k, M.FAIL_CLUSTER_EXAMPLE.get(k, ''))}" for k in keys]
    ax.set_yticks(y, labs, fontsize=9)
    ax.set_xlim(0, max(tot) * 1.32)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlabel(f"failed rounds (of {total}) · colour = how the agent's own session ended")
    ax.set_title("Why failing rounds fail — and whether the agent knew", pad=30)
    ax.legend(ncol=4, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              frameon=False)
    ax.grid(axis="y", visible=False)

    done = sum(grid[k]['agent said "done"'] for k in keys)
    _caption(fig,
             f"{done} of {total} failing rounds ended normally — the agent finished on its own "
             "terms and handed in something that does not work. That is a different problem "
             "from running out of room, and figure 03 merges the two.")
    save(fig, "23a_failure_by_end_reason.png")


_END_EXAMPLE = {
    "Ran out of budget": "context window exceeded, turn/step cap, output cap",
    "Stopped early, produced nothing": "no solution file, or an incomplete one",
}


# --------------------------------------------------------------------------- #
# 23b -- who owns each failure mode
# --------------------------------------------------------------------------- #
# 23a sizes the failure modes and says how the session ended. It does not say
# whether a mode is spread across the fleet or is one model's private problem.
# Same bars, same order, composition instead of size.
#
# Bars are normalised to 100 %. At raw scale the short bars cannot carry a
# label, and 23a already shows the sizes, so this figure answers composition
# only and the totals go on the tick labels.
#
# Models ran between 80 and 160 rounds, so a large share can just mean "ran more
# often". The correction for that is lift -- share of the cluster divided by
# share of all rounds, 1.0x neutral. It sits in the caption rather than on the
# segments, which already carry a count and a share.
#
# Layout copies 23a -- same bar height, same y labels, same legend above the
# title, same totals at the right end -- because the two slides run back to back
# and the reader should map bar to bar without re-reading the axis.
#
# Colours are 23a's four, in 23a's left-to-right order, so the pair reads as one
# slide. They mean rank here, not end reason, which is why the legend sits right
# above the bars on both figures.
_OWNER_TOP = 3
_OWNER_SHADE = [RED, AMBER, GREY]
_OWNER_REST = "#d8dde3"
_OWNER_LEGEND = ["heaviest model", "2nd", "3rd", "all other models"]


def _wrap_hyphen(s):
    """Break a model name into two lines at the hyphen nearest the middle."""
    cuts = [i for i, ch in enumerate(s) if ch == "-"]
    if not cuts:
        return s
    i = min(cuts, key=lambda i: abs(i - len(s) / 2))
    return s[:i] + "-\n" + s[i + 1:]


def fig_failure_owner_by_model():
    fails = [r for r in ROWS_LATEST if not r["passed"]]
    grid = collections.defaultdict(collections.Counter)
    for r in fails:
        c = _fail_cluster(r)
        if c == "Infra hiccup":
            continue
        grid[c][r["model"]] += 1

    exposure = collections.Counter(r["model"] for r in ROWS_LATEST)
    n_all = sum(exposure.values())
    keys = sorted(grid, key=lambda k: sum(grid[k].values()), reverse=True)
    total = len(fails)   # same denominator as 23a, so the right-hand numbers agree

    fig, ax = plt.subplots(figsize=(11.0, 0.78 * len(keys) + 2.4))
    y = np.arange(len(keys))[::-1]

    # Model names vary from 11 to 22 characters and segments from 1 % to 75 %,
    # so a fixed font size either overflows the narrow segments or wastes the
    # wide ones. Measure each label and shrink it until it fits, drop it if it
    # cannot fit at the floor.
    rend = fig.canvas.get_renderer()
    # Pixels per x unit. The axis runs to 132 to leave room for the totals on
    # the right, so a segment of `frac` percent is not `frac`% of the axes.
    px_per_unit = ax.get_window_extent(rend).width / 132

    def _fit(t, frac):
        room = 0.9 * frac * px_per_unit
        while t.get_window_extent(rend).width > room:
            s = t.get_fontsize() - 0.5
            if s < 6.0:
                t.remove()
                return False
            t.set_fontsize(s)
        return True

    for yy, k in zip(y, keys):
        tot = sum(grid[k].values())
        ranked = grid[k].most_common()
        left = 0.0
        for rank, (m, c) in enumerate(ranked[:_OWNER_TOP]):
            w = 100 * c / tot
            ax.barh(yy, w, left=left, height=0.55, color=_OWNER_SHADE[rank],
                    edgecolor="white", linewidth=0.5)
            # Name on top, count and share underneath.
            col = "white"
            mid = left + w / 2
            kw = dict(ha="center", va="center", color=col, fontweight="bold")
            if not _fit(ax.text(mid, yy + 0.09, _mlabel(m), fontsize=9, **kw), w):
                # A name like DeepSeek-V4-Flash-0731 does not fit a 14 % segment
                # on one line at a readable size. Break it at a hyphen instead
                # of shrinking it away.
                _fit(ax.text(mid, yy + 0.12, _wrap_hyphen(_mlabel(m)),
                             fontsize=8, linespacing=1.0, **kw), w)
            _fit(ax.text(mid, yy - 0.14, f"{c} · {w:.0f}%", ha="center",
                         va="center", fontsize=8, color=col, alpha=0.9), w)
            left += w
        rest = tot - sum(c for _, c in ranked[:_OWNER_TOP])
        if rest:
            n_rest = len(ranked) - _OWNER_TOP
            ax.barh(yy, 100 - left, left=left, height=0.55, color=_OWNER_REST,
                    edgecolor="white", linewidth=0.5)
            _fit(ax.text(left + (100 - left) / 2, yy,
                         f"{n_rest} other model{'s' if n_rest > 1 else ''} · {rest}",
                         ha="center", va="center", fontsize=8, color=INK),
                 100 - left)
        # Same right-hand readout as 23a, so the bar sizes carry over.
        ax.text(101.5, yy, f"{tot}  ({100 * tot / total:.0f}%)",
                va="center", fontsize=10, color=INK)

    # Legend proxies: the bars are drawn one segment at a time, so the rank
    # ramp has no labelled artist to pick up.
    for c, lab in zip(_OWNER_SHADE + [_OWNER_REST], _OWNER_LEGEND):
        ax.barh(0, 0, left=0, height=0.55, color=c, label=lab,
                edgecolor="white", linewidth=0.5)

    labs = [f"{k}\n{_END_EXAMPLE.get(k, M.FAIL_CLUSTER_EXAMPLE.get(k, ''))}" for k in keys]
    ax.set_yticks(y, labs, fontsize=9)
    ax.set_xlim(0, 132)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel("share of that failure mode · colour = the model's rank inside the mode")
    ax.set_title("Every failure mode has an owner", pad=30)
    ax.legend(ncol=4, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              frameon=False)
    ax.grid(axis="y", visible=False)

    top = {k: grid[k].most_common(1)[0] for k in keys}

    def _share(k):
        return 100 * top[k][1] / sum(grid[k].values())

    def _lift(k):
        return (top[k][1] / sum(grid[k].values())) / (exposure[top[k][0]] / n_all)

    owned = ["Software defect (build failed)", "Stopped early, produced nothing",
             "Ran out of budget"]
    _caption(fig,
             "Same bars as 23a, same order, composition instead of size: each one is split "
             "by the model the round ran on. The three heaviest models are named, the rest "
             "are pooled, and the right-hand number repeats the size from 23a.\n"
             f"Three of the {len(keys)} modes are one model's problem: Llama-3.3-70B causes "
             f"{_share('Software defect (build failed)'):.0f} % of the build failures, "
             f"gpt-oss-120b {_share('Stopped early, produced nothing'):.0f} % of the rounds "
             f"that hand in nothing, and Qwen3.6-35B-A3B-FP8 {_share('Ran out of budget'):.0f} % "
             "of the budget exhaustion. Wrong output is the one mode spread across the "
             "fleet, and it is the mode a better model has to fix rather than a better "
             "harness.\n"
             "Models ran between 80 and 160 rounds, so a share partly measures how often "
             "a model ran. Dividing each share by that model's share of all "
             f"{n_all} rounds removes that, and the three owners above come out at "
             + ", ".join(f"{_lift(k):.1f}x" for k in owned)
             + " their expected weight.")
    save(fig, "23b_failure_owner_by_model.png")


# --------------------------------------------------------------------------- #
# 24a / 24b -- adjusted pass rate, both directions of the same fit
# --------------------------------------------------------------------------- #
# Pooled pass@1 is fragile while the design is unbalanced: restricted to the
# models all four agents ran, dropping Kimi-K2.7-Code alone flips swe-mini and
# kimi. Win-counting is worse (it throws away cell size). One logistic fit
# uses every round, weights it by the cell it came from, and stays valid as
# cells are topped up to 40 rounds later -- rerun it, the intervals narrow,
# nothing else changes.
_FIT_CACHE = {}


def _fit():
    if "v" not in _FIT_CACHE:
        _FIT_CACHE["v"] = fit_adjusted(ROWS_LATEST)
    return _FIT_CACHE["v"]


def _adjusted_panel(axis, title, question, fname, figsize, labeller, hide=()):
    des, b, point, ci = _fit()
    names = des.agents if axis == "agent" else des.models
    # hide drops a row from the chart only. The fit still uses its rounds.
    keys = sorted([n for n in names if n not in hide], key=lambda n: -point[(axis, n)])
    pt = np.array([100 * point[(axis, k)] for k in keys])
    lo = np.array([100 * ci[(axis, k)][0] for k in keys])
    hi = np.array([100 * ci[(axis, k)][1] for k in keys])
    obs = collections.defaultdict(lambda: [0, 0])
    for r in ROWS_LATEST:
        c = obs[display_agent(r) if axis == "agent" else r["model"]]
        c[1] += 1
        c[0] += r["passed"]

    fig, ax = plt.subplots(figsize=figsize)
    _dotwhisker(ax, [labeller(k) for k in keys], pt, lo, hi, color=BLUE)
    for i, k in enumerate(keys):
        p, n = obs[k]
        ax.text(hi[i] + 1.5, len(keys) - 1 - i,
                f"{pt[i]:.1f}%  [{lo[i]:.0f}–{hi[i]:.0f}]   raw {100*p/n:.0f}%, n={n}",
                va="center", fontsize=9, color=INK)
    ax.set_xlim(-3, 118)
    ax.set_xlabel("adjusted pass rate (%), 95% bootstrap interval")
    ax.set_title(title)
    _caption(fig,
             question + "\nLogistic fit passed ~ agent + model + problem over all "
             f"{len(ROWS_LATEST)} rounds (ridge λ=1, 2,000 bootstrap resamples of the "
             f"{len(ROWS_LATEST)} rounds drawn independently). Each level is "
             "re-run counterfactually against the same mix of the other two factors, so no "
             "level is credited for having drawn an easier mix.\n"
             "Llama-3.3-70B shows a few percent rather than 0% because the ridge term refuses "
             "to assign an infinite penalty to a cell with zero passes. Where adjusted and raw "
             "differ a lot the level ran an unrepresentative mix"
             + (". claude pro is not shown: it only ever ran one model (Opus 4.8), so it has "
                "no agent effect that can be separated from that model."
                if "claude pro" in hide else
                " — claude pro only ever ran one "
                "model (Opus 4.8, 20 rounds), so its adjusted value is an extrapolation to models "
                "it never saw, which is what its wide interval is saying."))
    save(fig, fname)


def fig_adjusted_by_agent():
    _adjusted_panel(
        "agent",
        "Adjusted pass rate by agent",
        "Holding the model and the problem constant, does the agent change the odds of passing?",
        "24a_adjusted_by_agent.png", (10.4, 4.6), lambda k: k, hide=("claude pro",))


def fig_adjusted_by_model():
    _adjusted_panel(
        "model",
        "Adjusted pass rate by model",
        "Holding the agent and the problem constant, does the model change the odds of passing?",
        "24b_adjusted_by_model.png", (10.8, 7.6), _mlabel)


# --------------------------------------------------------------------------- #
# 26 -- are the four problems four copies of one test?
# --------------------------------------------------------------------------- #
def fig_problem_correlation():
    per = collections.defaultdict(lambda: [0, 0])
    for r in ROWS_LATEST:
        c = per[(r["agent"], r["model"], r["problem"])]
        c[1] += 1
        c[0] += r["passed"]
    pairs = collections.defaultdict(dict)
    for (a, m, p), (c, n) in per.items():
        pairs[(a, m)][p] = c / n
    probs = M.PROBLEMS
    full = [d for d in pairs.values() if all(p in d for p in probs)]
    n = len(probs)
    C = np.eye(n)
    for i in range(n):
        for j in range(n):
            xi = [d[probs[i]] for d in full]
            xj = [d[probs[j]] for d in full]
            C[i, j] = np.corrcoef(xi, xj)[0, 1]

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(C, cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=12,
                    color=INK if 0.25 < C[i, j] < 0.85 else "white", fontweight="bold")
    labs = [M.PROB_SHORT.get(p, p).splitlines()[0] for p in probs]
    ax.set_xticks(range(n), labs, fontsize=9)
    ax.set_yticks(range(n), labs, fontsize=9)
    ax.grid(False)
    ax.set_title("Do the four problems measure the same thing?")
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation of per-pairing pass rate")
    off = [C[i, j] for i in range(n) for j in range(n) if i != j]
    _caption(fig, axes_bottom_in=0,
             text=f"Pearson r across the {len(full)} pairings that ran all four problems. "
             f"Mean off-diagonal {np.mean(off):.2f}. The two RabbitCT variants are partly "
             "redundant with each other; tiled-gemm measures an ability the porting problems "
             "do not predict, so a fifth problem should be uncorrelated with all four.")
    save(fig, "26_problem_correlation.png")


# --------------------------------------------------------------------------- #
# 27a / 27b -- does spending more tokens buy a better result?
# --------------------------------------------------------------------------- #
# 16a/16b say what a solve costs. Neither says whether the spend earns anything.
# Two panels are needed because either one alone points the wrong way:
#
#   - Pooled by problem (23b's grouping, applied to tokens) passed rounds look
#     MORE expensive than failed ones -- LolaBunny 2.9 M vs 2.1 M, tiled-gemm
#     5.2 M vs 1.1 M. That is Llama-3.3-70B and gpt-oss dying in the first few
#     turns and dragging the failed median down. It measures "a round that got
#     going costs more than one that never started", not the spend question.
#   - Held at one model, the sign flips. Model is what sets the token scale
#     (0.18 M to 6.6 M per round), so 27b groups by model, not by problem.
#
# Rows the provider never metered and kimi_token_estimate could not reconstruct
# (tok_value() == 0, 28 of 1,132) drop out of the token axis only. The pass@1
# denominator stays the full round count, same as 16a/16b.
_TOP_BAND = 0.55        # 27a: the pass@1 floor for "the band that works"
_AUC_MIN_SIDE = 3       # 27b caption: min passes and fails for a cell to count


def _spearman(x, y):
    """Rank correlation, ties averaged. No scipy on the cluster."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def _auc(pos, neg):
    """P(a random passed round spent more than a random failed one), ties half.
    0.5 is no association; below 0.5 means the pricier round is the one that
    fails."""
    if not pos or not neg:
        return None
    c = sum(1 if a > b else 0.5 if a == b else 0 for a in pos for b in neg)
    return c / (len(pos) * len(neg))


def _tokens_by_model():
    """model -> {n, passed, tok, tn, est}. n counts every round; tok/tn only
    the rounds that carry a token count."""
    g = collections.defaultdict(lambda: {"n": 0, "passed": 0, "tok": 0.0,
                                         "tn": 0, "est": False})
    for r in ROWS_LATEST:
        e = g[r["model"]]
        e["n"] += 1
        e["passed"] += bool(r["passed"])
        t, estimated = M.tok_value(r)
        if t > 0:
            e["tok"] += t
            e["tn"] += 1
            e["est"] = e["est"] or estimated
    return g


def fig_pass_vs_tokens_model():
    g = _tokens_by_model()
    keys = [k for k in g if g[k]["tn"] > 0]
    x = [g[k]["tok"] / g[k]["tn"] / 1e6 for k in keys]
    y = [100 * g[k]["passed"] / g[k]["n"] for k in keys]
    # Mean tokens per ROUND, not 16b's tokens per solve: per-solve divides by
    # the pass count, so plotting it against pass@1 puts the same quantity on
    # both axes.
    rho_all = _spearman(x, y)
    live = [i for i in range(len(keys)) if y[i] > 25]
    rho_live = _spearman([x[i] for i in live], [y[i] for i in live])

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    # Band drawn across the point estimates of the top group, not their Wilson
    # bounds: Kimi-K3 runs on 12 rounds, so its interval reaches down past
    # gemma and a CI-spanned band would look like it includes a 33 % model.
    band = [i for i in range(len(keys)) if y[i] >= 100 * _TOP_BAND]
    gutter = max(x) * 1.14
    if band:
        ax.axhspan(min(y[i] for i in band), max(y[i] for i in band),
                   color=GREY, alpha=0.14, zorder=0)

    for i, k in enumerate(keys):
        lo, hi = wilson(g[k]["passed"], g[k]["n"])
        c = PURPLE if k == "claude-opus-4-8" else BLUE
        ax.vlines(x[i], 100 * lo, 100 * hi, color=c, linewidth=2.0, alpha=0.45, zorder=2)
        ax.scatter([x[i]], [y[i]], s=64, color=c, zorder=3)

    # All names live in a gutter to the right of the data, pushed apart
    # vertically and joined to their point by a leader. Labelling in place
    # does not work here: seven of the eleven models sit inside four pass-rate
    # points of each other, so a label next to one point lands on top of the
    # next point's marker.
    order = sorted(range(len(keys)), key=lambda i: -y[i])
    ly = {}
    prev = None
    for i in order:
        v = y[i] if prev is None else min(y[i], prev - 4.6)
        ly[i] = v
        prev = v
    for i, k in enumerate(keys):
        ax.plot([x[i] + max(x) * 0.02, gutter - max(x) * 0.015], [y[i], ly[i]],
                color=GREY, linewidth=0.7, alpha=0.75, zorder=1)
        ax.annotate(f"{_mlabel(k)}  {y[i]:.0f}%", (gutter, ly[i]), fontsize=9.5,
                    color=PURPLE if k == "claude-opus-4-8" else INK,
                    va="center", ha="left")

    ax.set_xlim(0, max(x) * 1.70)
    ax.set_ylim(-6, 100)
    ax.set_xlabel("mean tokens per round, millions (every round, passed or failed)")
    ax.set_ylabel("pass@1 (%)")
    ax.set_title("Pass rate against tokens spent, by model")
    _caption(fig, axes_bottom_in=0,
             text="Bars are 95 % Wilson intervals on pass@1. The upward slope is "
             "Llama-3.3-70B and gpt-oss-120b, which fail early enough that they never "
             f"get to spend: rank correlation is {rho_all:.2f} over all {len(keys)} "
             f"models but {rho_live:.2f} over the {len(live)} above 25 %. Inside the "
             "shaded band nothing separates a 1.4 M-token round from a 6.6 M-token "
             "one, and the cheapest model there also has the highest rate. Claude "
             "Opus 4.8 (purple) is the one closed-source model, run through the direct "
             "Anthropic API; token counts for every other model include estimated "
             "kimi-agent rows.")
    save(fig, "27a_pass_vs_tokens_by_model.png")


def fig_tokens_passed_vs_failed_model():
    g = collections.defaultdict(lambda: {True: [], False: []})
    for r in ROWS_LATEST:
        t, _ = M.tok_value(r)
        if t > 0:
            g[r["model"]][bool(r["passed"])].append(t)
    # A model with no passes has no pair to draw (Llama-3.3-70B, 0 of 120).
    keys = [k for k, v in g.items() if v[True] and v[False]]
    med = {k: (st.median(g[k][True]) / 1e6, st.median(g[k][False]) / 1e6) for k in keys}
    keys.sort(key=lambda k: med[k][1] / med[k][0], reverse=True)

    # One AUC per (agent, model, problem) cell: the caption's single number,
    # and the only view that holds the problem constant as well as the model.
    aucs = []
    cells = collections.defaultdict(lambda: {True: [], False: []})
    for r in ROWS_LATEST:
        t, _ = M.tok_value(r)
        if t > 0:
            cells[(r["agent"], r["model"], r["problem"])][bool(r["passed"])].append(t)
    for v in cells.values():
        if len(v[True]) >= _AUC_MIN_SIDE and len(v[False]) >= _AUC_MIN_SIDE:
            aucs.append(_auc(v[True], v[False]))
    mean_auc = sum(aucs) / len(aucs)
    below = sum(1 for a in aucs if a < 0.5)

    xmax = max(m[1] for m in med.values())
    fig, ax = plt.subplots(figsize=(10.4, 0.62 * len(keys) + 2.0))
    y = np.arange(len(keys))[::-1]
    h = 0.34
    for off, passed, color, name in ((h / 2, True, GREEN, "passed"),
                                     (-h / 2, False, RED, "failed")):
        w = [med[k][0 if passed else 1] for k in keys]
        ax.barh(y + off, w, height=h, color=color, label=name)
        for yy, ww, k in zip(y, w, keys):
            ax.text(ww + xmax * 0.015, yy + off,
                    f"{ww:.2f} M   n={len(g[k][passed])}", va="center",
                    fontsize=9, color=INK)

    ax.set_yticks(y, [_mlabel(k) for k in keys], fontsize=10)
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.set_xlim(0, xmax * 1.22)
    ax.set_xlabel("median tokens per round, millions")
    ax.set_title("Tokens per round in passed and failed rounds, by model")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.grid(axis="y", visible=False)
    _caption(fig, axes_bottom_in=0,
             text=f"Failed rounds cost more than passed rounds in "
             f"{sum(1 for k in keys if med[k][1] > med[k][0])} of the {len(keys)} "
             "models, so within one model the pricier round is the one more likely to "
             "fail. Ranking rounds by token count inside each (agent, model, problem) "
             f"cell says the same: mean AUC {mean_auc:.2f} across the {len(aucs)} cells "
             f"with at least {_AUC_MIN_SIDE} passes and {_AUC_MIN_SIDE} fails, {below} "
             "of them below the 0.50 no-association line. Grouped by model rather than "
             "by problem because model is what sets the token scale. Llama-3.3-70B is "
             "absent: 0 of 120 rounds passed, so it has no passed bar.")
    save(fig, "27b_tokens_passed_vs_failed_by_model.png")

# --------------------------------------------------------------------------- #
# 27c -- 27a with the x-axis priced
# --------------------------------------------------------------------------- #
# Same rounds, same y-axis, same layout as 27a. The x-axis becomes mean US
# dollars per round: input and output are metered separately and charged at the
# midpoint of the OpenRouter range (OPENROUTER_USD, defined under 29).
#
# Per ROUND, not per solve. E($) from 29 is dollars per round divided by pass@1,
# and pass@1 is this figure's y-axis, so a point far to the right would not say
# whether the model spent more or passed less. Same reason 27a's x is per round
# rather than 16b's per solve. 29 is the figure that charges for the retries.
#
# Log x: dollars span 127x across the models (0.06 to 7.39) where tokens span
# 5.7x, so a linear axis stacks half the models on the spine.
def _usd_by_model():
    """model -> {n, passed, usd, tn, est}. n counts every round, usd/tn only
    the rounds that carry a token count, same split as _tokens_by_model."""
    g = collections.defaultdict(lambda: {"n": 0, "passed": 0, "usd": 0.0,
                                         "tn": 0, "est": False})
    for r in ROWS_LATEST:
        e = g[r["model"]]
        e["n"] += 1
        e["passed"] += bool(r["passed"])
        i, o, estimated = M.io_split(r)
        if i + o > 0 and r["model"] in OPENROUTER_USD:
            il, ih, ol, oh = OPENROUTER_USD[r["model"]]
            e["usd"] += i / 1e6 * (il + ih) / 2 + o / 1e6 * (ol + oh) / 2
            e["tn"] += 1
            e["est"] = e["est"] or estimated
    return g


def fig_pass_vs_dollars_model():
    g = _usd_by_model()
    keys = [k for k in g if g[k]["tn"] > 0]
    x = [g[k]["usd"] / g[k]["tn"] for k in keys]
    y = [100 * g[k]["passed"] / g[k]["n"] for k in keys]
    rho_all = _spearman(x, y)
    live = [i for i in range(len(keys)) if y[i] > 25]
    rho_live = _spearman([x[i] for i in live], [y[i] for i in live])

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    # Gutter and leader offsets are multiplicative here, not additive: the axis
    # is logarithmic, so a fixed dollar offset is a different gap at each x.
    lo_x, hi_x = min(x), max(x)
    gutter = hi_x * 1.5

    # Shading only, no note. 27a's corner note does not fit here: the gutter
    # labels already reach the top of the panel, so the band is described in
    # the caption instead.
    band = [i for i in range(len(keys)) if y[i] >= 100 * _TOP_BAND]
    if band:
        ax.axhspan(min(y[i] for i in band), max(y[i] for i in band),
                   color=GREY, alpha=0.14, zorder=0)

    for i, k in enumerate(keys):
        lo, hi = wilson(g[k]["passed"], g[k]["n"])
        c = PURPLE if k == "claude-opus-4-8" else BLUE
        ax.vlines(x[i], 100 * lo, 100 * hi, color=c, linewidth=2.0, alpha=0.45, zorder=2)
        ax.scatter([x[i]], [y[i]], s=64, color=c, zorder=3)

    order = sorted(range(len(keys)), key=lambda i: -y[i])
    ly = {}
    prev = None
    for i in order:
        v = y[i] if prev is None else min(y[i], prev - 4.6)
        ly[i] = v
        prev = v
    for i, k in enumerate(keys):
        ax.plot([x[i] * 1.09, gutter * 0.94], [y[i], ly[i]],
                color=GREY, linewidth=0.7, alpha=0.75, zorder=1)
        ax.annotate(f"{_mlabel(k)}  \\${x[i]:.2f}  ·  {y[i]:.0f}%", (gutter, ly[i]),
                    fontsize=9.5, color=PURPLE if k == "claude-opus-4-8" else INK,
                    va="center", ha="left")

    ax.set_xscale("log")
    ax.set_xlim(lo_x * 0.55, hi_x * 22)
    ax.set_ylim(-6, 100)
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2.5, 5])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"${v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("mean US dollars per round at OpenRouter list price, log scale "
                  "(every round, passed or failed)")
    ax.set_ylabel("pass@1 (%)")
    ax.set_title("Pass rate against list price per round, by model")

    cheap = min(band, key=lambda i: x[i]) if band else None
    dear = max(band, key=lambda i: x[i]) if band else None
    _caption(fig, axes_bottom_in=0,
             text="27a with the x-axis priced: the same rounds and the same pass@1, "
             "but input and output tokens charged separately at the midpoint of the "
             "OpenRouter range. Dollars per ROUND, won or lost — 29 is the figure that "
             "also charges for the retries. Bars are 95 % Wilson intervals on pass@1.\n"
             "Pricing reverses 27a's reading. On tokens, spend and pass rate come apart "
             "inside the shaded band; on dollars they do not, because price tracks model "
             f"class: rank correlation rises from 0.31 to {rho_all:.2f} over all "
             f"{len(keys)} models and from 0.05 to {rho_live:.2f} over the {len(live)} "
             "above 25 %. The shaded band is where the money is decided: "
             f"{len(band)} models within "
             f"{max(y[i] for i in band) - min(y[i] for i in band):.0f} points of each "
             f"other, {_mlabel(keys[dear])} at \\${x[dear]:.2f} per round against "
             f"{_mlabel(keys[cheap])} at \\${x[cheap]:.2f}, "
             f"{x[dear] / x[cheap]:.0f} times the price for "
             f"{y[dear] - y[cheap]:.0f} points of pass@1.\n"
             "Nothing was billed at these rates: every open model ran on A40s and "
             "Claude Opus 4.8 (purple) on a subscription. These are OpenRouter list "
             "prices as of 2026-09-04 for the same models, so the axis is what a round "
             "would be expected to cost renting the model instead of hosting it. Token "
             "counts for every model except Opus include estimated kimi-agent rows.")
    save(fig, "27c_pass_vs_dollars_by_model.png")

# --------------------------------------------------------------------------- #
# 28 -- expected token cost to reach one solve, E(k), by model
# --------------------------------------------------------------------------- #
# 16d charges a model only for the rounds it won: tokens summed over solved
# rounds, divided by the number of solves. A model that passes one round in
# three therefore looks cheap, because the two failed attempts are free.
#
# ParEval's E(k) closes that hole -- see README.md:188. Expected number of
# attempts to land a pass (1 / pass@1) times the average cost of one attempt,
# where the average runs over EVERY round, won or lost:
#
#     E(k) = mean tokens per round / pass@1
#
# Same bars, same prompt/completion split and same helpers as 16d
# (M.io_split, M._cost_split_bars, M._cost_split_legend), so the two figures
# can be read side by side; only the quantity on the x-axis changes.
#
# Lives here rather than next to 16d because it needs pass@1 and a caption,
# and because it answers the same question as 27a/27b: what the spend buys.
def _ordinal(n):
    return {1: "", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(n, f"{n}th")


_EK_LABEL = {"gemma-4-31B-it-FP8-block": "gemma-4-31B-it (FP8)",
             "claude-opus-4-8": "Claude Opus 4.8 (claude pro)"}


def _ek_by_model():
    """Per-model E(k) pieces, shared by 28 and 29.

    Two denominators on purpose. pass@1 counts every round in ROWS_LATEST, so
    it matches 17b / 20b / 27a exactly. The token mean can only run over the
    rounds that carry a count (28 of 1,132 do not, and 27 of those 28 are
    failures), which nudges E(k) up slightly rather than down.

    Returns (acc, keys, g, rate, per_round, never). g is shaped for
    M._cost_split_bars with n=1: inp/out already ARE the expected totals, not
    per-round means waiting to be divided."""
    acc = collections.defaultdict(lambda: {"inp": 0.0, "out": 0.0, "tn": 0,
                                           "n": 0, "passed": 0, "est": False,
                                           "won": 0.0, "wn": 0})
    for r in ROWS_LATEST:
        e = acc[r["model"]]
        e["n"] += 1
        e["passed"] += bool(r["passed"])
        i, o, estimated = M.io_split(r)
        if i + o > 0:
            e["inp"] += i
            e["out"] += o
            e["tn"] += 1
            e["est"] = e["est"] or estimated
            if r["passed"]:
                # won/wn reproduce 16d's own bar exactly -- tokens summed over
                # solved rounds only, over the number of solves -- so 28's
                # caption can name the model the retry charge moves furthest.
                e["won"] += i + o
                e["wn"] += 1

    # A model that never passes has no expected number of attempts -- 1/0.
    # Llama-3.3-70B (0 of 120) is the only one, and it is named in the caption
    # rather than drawn as an off-scale bar.
    never = [k for k, e in acc.items() if e["passed"] == 0 and e["tn"] > 0]
    keys = [k for k, e in acc.items() if e["passed"] > 0 and e["tn"] > 0]

    g, per_round, rate = {}, {}, {}
    for k in keys:
        e = acc[k]
        rate[k] = e["passed"] / e["n"]
        per_round[k] = (e["inp"] + e["out"]) / e["tn"]
        g[k] = {"inp": e["inp"] / e["tn"] / rate[k],
                "out": e["out"] / e["tn"] / rate[k],
                "n": 1, "est": e["est"]}
    keys.sort(key=lambda k: g[k]["inp"] + g[k]["out"])
    return acc, keys, g, rate, per_round, never


def fig_expected_token_cost_model():
    acc, keys, g, rate, per_round, never = _ek_by_model()

    labs = [f"{_EK_LABEL.get(k) or M.MODEL_SHORT.get(k, k).splitlines()[0]}\n"
            f"{per_round[k] / 1e6:.1f} M per round  ÷  {100 * rate[k]:.0f} % pass@1"
            for k in keys]

    fig, ax = plt.subplots(figsize=(9.6, 0.66 * len(keys) + 2.6))
    M._cost_split_bars(ax, keys, labs, g,
                       "E(κ): expected million tokens to reach one solve, "
                       "pooled across agents")
    ax.set_xlim(0, (g[keys[-1]]["inp"] + g[keys[-1]]["out"]) / 1e6 * 1.45)
    ax.tick_params(axis="y", labelsize=9)
    M._cost_split_legend(ax, g, keys, est_label="includes estimated (kimi) rows",
                         ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.095))
    fig.subplots_adjust(bottom=0.14)
    ax.set_title("Expected token cost to reach one solve, by model")

    ek = {k: (g[k]["inp"] + g[k]["out"]) / 1e6 for k in keys}
    d16 = {k: acc[k]["won"] / acc[k]["wn"] / 1e6 for k in keys}
    cheap = min(keys, key=lambda k: ek[k])
    moved = max(keys, key=lambda k: ek[k] / d16[k])
    _caption(fig, axes_bottom_in=0,
             text="16d charges a model only for the rounds it won, so one that passes "
             "rarely still looks cheap there. E(κ) charges it for the attempts too, and "
             f"the ranking changes hands. {_mlabel(moved)} moves furthest: the cheapest "
             f"bar in 16d at {1e3 * d16[moved]:.0f} K per solve, but it only passes "
             f"{100 * rate[moved]:.0f} % of the time, so landing one solve is expected "
             f"to cost {ek[moved]:.1f} M -- {ek[moved] / d16[moved]:.0f} times its 16d "
             f"figure. {_mlabel(cheap)} goes the other way: "
             f"{_ordinal(sorted(d16, key=lambda k: d16[k]).index(cheap) + 1)}-cheapest "
             f"in 16d at {d16[cheap]:.1f} M, cheapest here at {ek[cheap]:.1f} M against "
             f"{ek[max(keys, key=lambda k: ek[k])]:.1f} M for the most expensive. The "
             "prompt/completion split is unchanged from 16d: retries scale both sides "
             "equally. "
             + (f"{', '.join(_mlabel(k) for k in never)} cannot be drawn -- 0 passes, so "
                "no finite number of attempts is expected to produce one." if never else ""))
    save(fig, "28_expected_token_cost_by_model.png")

# --------------------------------------------------------------------------- #
# 29 -- expected DOLLAR cost to reach one solve, at OpenRouter list prices
# --------------------------------------------------------------------------- #
# 28 counts tokens, which treats every token as worth the same. They are not:
# per million tokens the models here span 0.04 to 5.00 dollars on input and
# 0.17 to 25.00 on output, a 100x+ spread that no token count can show. This
# figure prices 28's expected token counts:
#
#     E($) = E(k)_input x price_in + E(k)_output x price_out
#
# Nothing was actually billed at these rates. Every open model here ran on
# A40s and Opus ran on a Claude subscription; these are OpenRouter list prices
# for the same model, i.e. what one solve WOULD be expected to cost renting the
# model instead of hosting it. Prices are as supplied on 2026-09-04.
#
# (input_low, input_high, output_low, output_high), USD per million tokens.
# A range is several OpenRouter providers quoting the same model at different
# rates; the bar is drawn at the midpoint and the whisker spans the range.
OPENROUTER_USD = {
    "Qwen3.8-27B": (0.40, 0.45, 2.55, 3.20),
    "Qwen3.6-35B-A3B-FP8": (0.14, 0.17, 1.00, 1.10),
    "DeepSeek-V4-Flash": (0.14, 0.14, 0.28, 0.28),
    "Qwen3.6-27B": (0.20, 0.60, 1.56, 3.60),
    "DeepSeek-V4-Flash-0731": (0.09, 0.14, 0.18, 0.28),
    "Kimi-K2.7-Code": (0.61, 0.95, 3.07, 4.00),
    "Kimi-K3": (3.00, 3.00, 15.00, 15.00),
    "claude-opus-4-8": (5.00, 5.00, 25.00, 25.00),
    "gpt-oss-120b": (0.04, 0.05, 0.17, 0.20),
    "gemma-4-31B-it-FP8-block": (0.10, 0.13, 0.34, 0.38),
    "Llama-3.3-70B-Instruct": (0.10, 0.10, 0.32, 0.32),
}


def fig_expected_dollar_cost_model():
    acc, keys, g, rate, per_round, never = _ek_by_model()
    keys = [k for k in keys if k in OPENROUTER_USD]

    usd = {}
    for k in keys:
        il, ih, ol, oh = OPENROUTER_USD[k]
        i, o = g[k]["inp"] / 1e6, g[k]["out"] / 1e6
        usd[k] = {"in": i * (il + ih) / 2, "out": o * (ol + oh) / 2,
                  "lo": i * il + o * ol, "hi": i * ih + o * oh}
    keys.sort(key=lambda k: usd[k]["in"] + usd[k]["out"])
    tot = {k: usd[k]["in"] + usd[k]["out"] for k in keys}

    fig, ax = plt.subplots(figsize=(9.8, 0.66 * len(keys) + 2.6))
    y = np.arange(len(keys))
    left = np.zeros(len(keys))
    for name, color, w in (("input", M.IN_COLOR, np.array([usd[k]["in"] for k in keys])),
                           ("output", M.OUT_COLOR, np.array([usd[k]["out"] for k in keys]))):
        bars = ax.barh(y, w, left=left, height=0.55, color=color, label=name,
                       edgecolor="white", linewidth=0.6)
        for b, k in zip(bars, keys):
            if g[k]["est"]:
                b.set_hatch("////")
        left += w

    # Whisker only where several providers quote different rates; a single
    # quoted price would draw a zero-width bracket that reads as a glitch.
    for yy, k in zip(y, keys):
        if usd[k]["hi"] - usd[k]["lo"] > 0.005:
            ax.hlines(yy, usd[k]["lo"], usd[k]["hi"], color=INK, linewidth=1.3, zorder=4)
            ax.vlines([usd[k]["lo"], usd[k]["hi"]], yy - 0.13, yy + 0.13,
                      color=INK, linewidth=1.3, zorder=4)

    for yy, k in zip(y, keys):
        share_in = 100 * usd[k]["in"] / tot[k]
        rng = ("" if usd[k]["hi"] - usd[k]["lo"] <= 0.005
               else f"   [{usd[k]['lo']:.2f}–{usd[k]['hi']:.2f}]")
        ax.text(max(tot[k], usd[k]["hi"]) + max(tot.values()) * 0.02, yy,
                f"\\${tot[k]:.2f}:  \\${usd[k]['in']:.2f} ({share_in:.0f}%) in"
                f"  /  \\${usd[k]['out']:.2f} out{rng}",
                va="center", fontsize=9.5, color=INK)

    il_ih = lambda k: OPENROUTER_USD[k]
    ax.set_yticks(y, [
        f"{_EK_LABEL.get(k) or M.MODEL_SHORT.get(k, k).splitlines()[0]}\n"
        f"{(g[k]['inp'] + g[k]['out']) / 1e6:.1f} M tokens  ×  "
        f"\\${sum(il_ih(k)[:2]) / 2:.2f} in / \\${sum(il_ih(k)[2:]) / 2:.2f} out per M"
        for k in keys], fontsize=9)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlim(0, max(tot.values()) * 1.45)
    ax.set_xlabel("E($): expected US dollars to reach one solve, at OpenRouter list prices")
    ax.grid(axis="y", visible=False)
    ax.set_title("Expected dollar cost to reach one solve, by model")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=M.IN_COLOR),
               plt.Rectangle((0, 0), 1, 1, facecolor=M.OUT_COLOR),
               plt.Line2D([], [], color=INK, linewidth=1.3),
               plt.Rectangle((0, 0), 1, 1, facecolor=GREY, hatch="////")]
    ax.legend(handles=handles,
              labels=["input spend", "output spend",
                      "range across providers", "includes estimated (kimi) rows"],
              frameon=False, fontsize=9, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.095))
    fig.subplots_adjust(bottom=0.14)

    tok_cheap = min(keys, key=lambda k: g[k]["inp"] + g[k]["out"])
    usd_cheap = keys[0]
    out_share = 100 * sum(usd[k]["out"] for k in keys) / sum(tot.values())
    tok_out_share = 100 * sum(g[k]["out"] for k in keys) / sum(g[k]["inp"] + g[k]["out"]
                                                              for k in keys)
    _caption(fig, axes_bottom_in=0,
             text="28's token counts, priced. Rented rather than self-hosted, the token "
             f"ranking inverts: {_mlabel(tok_cheap)} needs the fewest tokens of any model "
             f"({(g[tok_cheap]['inp'] + g[tok_cheap]['out']) / 1e6:.1f} M) and is the most "
             f"expensive to reach a solve with, at \\${tot[tok_cheap]:.2f}, because it "
             f"charges \\${OPENROUTER_USD[tok_cheap][0]:.2f} per million input tokens "
             f"against \\${OPENROUTER_USD[usd_cheap][0]:.2f} for {_mlabel(usd_cheap)}, "
             f"which lands a solve for \\${tot[usd_cheap]:.2f}. Output also stops being a "
             f"rounding error: {tok_out_share:.0f} % of the tokens but {out_share:.0f} % "
             "of the spend, because completion is priced 3 to 8 times input. Nothing was "
             "billed at these rates — every open model ran on A40s and Opus on a "
             "subscription. These are OpenRouter list prices as of 2026-09-04 for the "
             "same models, and a range is several providers quoting one model "
             "differently.")
    save(fig, "29_expected_dollar_cost_by_model.png")

# --------------------------------------------------------------------------- #
# 22d -- recoverable failure vs capability gap, per model
# --------------------------------------------------------------------------- #
# 22c answers the same question per pairing, but 35 rows x 5 segments is more
# chart than the point needs. Aggregating to the model collapses it to 10 rows
# x 3 segments that sum to 100, and the split *is* the title. Aggregating to
# the agent instead is nearly flat (43.5 / 43.2 / 39.6 / 32.8 %), which is why
# only the model cut gets a figure -- see the caption.
_REC_MIN_N = 5


def _recovery_table(k=5, by="model"):
    """key -> (pass@1, recovered by k, never solved, recovery share, n).

    Cell is (agent, model, problem), same unit as _passk_table, but kept per
    cell instead of per pairing so a model is not dropped for the agents it
    never ran. Cells with fewer than k rounds cannot support a pass@k estimate
    and are skipped -- that is what removes Kimi-K3 (12 rounds, 3 per problem).

    by="model" groups the cells by model, by="agent" by agent. Both sides read
    off the identical cells, so 22f compares like with like.
    """
    cell = collections.defaultdict(lambda: [0, 0])
    for r in ROWS_LATEST:
        c = cell[(display_agent(r), r["model"], r["problem"])]
        c[1] += 1
        c[0] += r["passed"]

    idx = 0 if by == "agent" else 1
    per = collections.defaultdict(list)
    for key, (c, n) in cell.items():
        if n >= k:
            per[key[idx]].append((100 * pass_at_k(n, c, 1), 100 * pass_at_k(n, c, k), n))

    out = {}
    for m, v in per.items():
        p1 = st.mean(x[0] for x in v)
        pk = st.mean(x[1] for x in v)
        # Undefined at pass@1 = 100 (no failures to recover); no cell hits it.
        share = 100 * (pk - p1) / (100 - p1) if p1 < 99.5 else float("nan")
        out[m] = (p1, pk - p1, 100 - pk, share, sum(x[2] for x in v))
    return out


def fig_recovery_by_model(k=5):
    tab = _recovery_table(k)
    keys = sorted(tab, key=lambda m: (-tab[m][3], -tab[m][0]))
    p1 = np.array([tab[m][0] for m in keys])
    rec = np.array([tab[m][1] for m in keys])
    gap = np.array([tab[m][2] for m in keys])

    fig, ax = plt.subplots(figsize=(11.0, 0.46 * len(keys) + 2.6))
    y = np.arange(len(keys))[::-1]
    segs = [(p1, BLUE, "solved on the first try"),
            (rec, GREEN, f"recovered by attempt {k}"),
            (gap, RED, f"never solved in {k} attempts")]
    left = np.zeros(len(keys))
    for val, col, lab in segs:
        ax.barh(y, val, height=0.62, left=left, color=col, label=lab)
        for i, v in enumerate(val):
            if v >= 6:
                ax.text(left[i] + v / 2, y[i], f"{v:.0f}", ha="center", va="center",
                        fontsize=8.5, color="white")
        left += val

    for i, m in enumerate(keys):
        share, n = tab[m][3], tab[m][4]
        txt = "—" if share != share else f"{share:.0f}%"
        ax.text(105, y[i], txt, va="center", ha="right", fontsize=9.5, color=INK)
        ax.text(108, y[i], f"n={n}", va="center", ha="left", fontsize=8, color=GREY)
    ax.text(105, len(keys) - 0.62, "recovery share", va="bottom", ha="right",
            fontsize=8.5, color=INK)

    ax.set_yticks(y, [_mlabel(m) for m in keys], fontsize=9.5)
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.set_xlim(0, 124)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("share of rounds (%)")
    ax.set_title("How much failure is variance, not a capability gap", pad=26)
    ax.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.002),
              frameon=False, ncol=3)
    ax.grid(axis="y", visible=False)
    _caption(fig,
             "Same pass@k estimator as 22c (1 - C(n-c,k)/C(n,k) per agent-model-problem "
             f"cell, k={k}), averaged over a model's cells, so the three segments sum to 100. "
             "Recovery share is the green block divided by everything the model failed on "
             "the first try, (pass@k - pass@1) / (100 - pass@1): of the rounds it missed "
             "immediately, the fraction it reached within "
             f"{k} attempts. High share means the capability is there and the first solution "
             "is not; low share means retrying does not reach it.\n"
             "Reading: Qwen3.8-27B and Qwen3.6-27B post the same 62.5 % first-try rate and "
             "split 50 % against 82 %, so pass@1 alone hides which failures a retry budget "
             "would buy back. Qwen3.6-35B-A3B-FP8 sits near the bottom of 24b yet recovers a "
             "larger share than Kimi-K2.7-Code and gemma, both of which beat it on pass@1.\n"
             "The same cut by agent is nearly flat — swe-mini 43.5 %, claude 43.2 %, pi "
             "39.6 %, kimi 32.8 % — against a 0 to 91 % spread across models, so "
             "recoverability tracks the model rather than the harness.\n"
             "Claude Opus 4.8 is 20 rounds under one agent, the same thin cell 24b "
             "flags. Llama-3.3-70B's 0 % is not thin — it passed 0 of 120.")
    save(fig, "22d_recovery_by_model.png")


# --------------------------------------------------------------------------- #
# 22e -- pass@1 against recovery share
# --------------------------------------------------------------------------- #
# 22d ranks models on recovery share. This one shows the ranking is not a
# restatement of the pass@1 ranking: the two axes are only loosely coupled, and
# the Qwen pair proves it by sitting on the same x with 32 pp between them.
# Ten fixed points, so the label offsets are hand-placed rather than solved.
_SCATTER_NUDGE = {                      # model -> (dx, dy, ha, va)
    "DeepSeek-V4-Flash-0731":   (2.2, 0.0, "left", "center"),
    "Qwen3.6-27B":              (0.0, 3.2, "center", "bottom"),
    "DeepSeek-V4-Flash":        (-2.2, -0.4, "right", "center"),
    "Qwen3.8-27B":              (2.2, 0.0, "left", "center"),
    "Qwen3.6-35B-A3B-FP8":      (0.0, 3.2, "center", "bottom"),
    "Kimi-K2.7-Code":           (2.2, 0.0, "left", "center"),
    "gemma-4-31B-it-FP8-block": (0.0, -3.2, "center", "top"),
    "gpt-oss-120b":             (2.2, 0.0, "left", "center"),
    "claude-opus-4-8":          (2.2, 0.0, "left", "center"),
    "Llama-3.3-70B-Instruct":   (2.2, 0.0, "left", "center"),
}


def fig_recovery_vs_pass1(k=5):
    tab = _recovery_table(k)

    fig, ax = plt.subplots(figsize=(10.4, 6.4))

    for m, (p1, _rec, _gap, share, n) in tab.items():
        if share != share:
            continue
        # Area tracks rounds behind the point, 20 to 160 across the panel.
        ax.scatter([p1], [share], s=14 + 1.2 * n, color=BLUE, alpha=0.75,
                   edgecolor="white", linewidth=1.0, zorder=3)
        dx, dy, ha, va = _SCATTER_NUDGE.get(m, (2.2, 0.0, "left", "center"))
        ax.text(p1 + dx, share + dy, _mlabel(m), ha=ha, va=va,
                fontsize=9, color=INK, zorder=4)

    ax.set_xlim(-6, 88)
    ax.set_ylim(-9, 102)
    ax.set_xlabel("solved on the first try (%), pass@1")
    ax.set_ylabel(f"recovery share (%): of first-try failures,\nthe share reached by attempt {k}")
    _caption(fig,
             "One point per model, the same two numbers 22d ranks on. Marker area is the "
             "rounds behind the point (20 to 160), so the small markers move most under "
             "resampling.\n"
             "Up the y-axis a model's failures are timing and a retry budget buys them back; "
             "along the x-axis it succeeds immediately. The two are close to independent here, "
             "which is the whole reason 22d is not a restatement of 24b: "
             "Qwen3.6-35B-A3B-FP8 clears Kimi-K2.7-Code on recovery from a lower pass@1.\n"
             "Claude Opus 4.8 and Llama-3.3-70B both land on y=0, for opposite reasons: Opus "
             "solved 75 % first try and never recovered the rest in 20 rounds under one agent, "
             "Llama passed 0 of 120.")
    save(fig, "22e_recovery_vs_pass1.png")


# --------------------------------------------------------------------------- #
# 22f -- the same recovery share, cut by model and by agent
# --------------------------------------------------------------------------- #
# 22d claims recoverability is a model property. This is the evidence: identical
# cells, identical arithmetic, grouped two ways. The model cut spans the whole
# axis, the agent cut sits inside an 11 pp band.
#
# claude pro is drawn grey and left out of the agent span on purpose. It is one
# model on four cells (Opus, 20 rounds), the same rounds as the Opus dot in the
# top panel, so its 0 % is a model effect reappearing rather than a harness
# effect. The other four agents each ran the same nine models.
def _span_pp(tab, keys):
    """Span of recovery share across keys, in percentage points."""
    v = [tab[k][3] for k in keys]
    return f"{max(v) - min(v):.0f}"


def fig_recovery_model_vs_agent(k=5):
    mods = _recovery_table(k, "model")
    ags = _recovery_table(k, "agent")
    mk = sorted(mods, key=lambda m: -mods[m][3])
    ak = sorted([a for a in ags if a != "claude pro"], key=lambda a: -ags[a][3])
    thin = [a for a in ags if a == "claude pro"]

    fig, axes = plt.subplots(
        2, 1, figsize=(10.6, 6.9), sharex=True,
        gridspec_kw={"height_ratios": [len(mk), len(ak) + len(thin) + 0.6],
                     "hspace": 0.30})

    def panel(ax, keys, labels, span_keys, tab, title, color):
        y = np.arange(len(keys))[::-1]
        lo = min(tab[s][3] for s in span_keys)
        hi = max(tab[s][3] for s in span_keys)
        ax.axvspan(lo, hi, color=color, alpha=0.10, zorder=0)
        for yy, kk, lab in zip(y, keys, labels):
            c = color if kk in span_keys else GREY
            ax.hlines(yy, 0, tab[kk][3], color=c, linewidth=1.6, alpha=0.45, zorder=2)
            ax.scatter([tab[kk][3]], [yy], s=58, color=c, zorder=3)
            ax.text(tab[kk][3] + 2.0, yy, f"{tab[kk][3]:.0f}", va="center",
                    ha="left", fontsize=9, color=c)
        ax.set_yticks(y, labels, fontsize=9.5)
        ax.set_ylim(-0.8, len(keys) - 0.2)
        ax.set_title(title, fontsize=12, pad=8, loc="left")
        ax.grid(axis="y", visible=False)
        return lo, hi

    mlo, mhi = panel(ax=axes[0], keys=mk, labels=[_mlabel(m) for m in mk],
                     span_keys=set(mk), tab=mods,
                     title=f"by model — spans {_span_pp(mods, mk)} pp", color=BLUE)
    alo, ahi = panel(ax=axes[1], keys=ak + thin, labels=ak + thin,
                     span_keys=set(ak), tab=ags,
                     title=f"by agent — spans {_span_pp(ags, ak)} pp", color=AMBER)

    for ax, lo, hi, col in ((axes[0], mlo, mhi, BLUE), (axes[1], alo, ahi, AMBER)):
        ax.set_xlim(0, 108)
        ax.annotate("", xy=(lo, -0.55), xytext=(hi, -0.55),
                    arrowprops=dict(arrowstyle="<->", color=col, linewidth=1.2))
    axes[1].set_xlabel(f"recovery share (%): of the rounds failed on the first try,"
                       f" the share reached by attempt {k}")
    fig.suptitle("Recoverability is a model property, not a harness property",
                 fontsize=15, fontweight="bold", y=0.965)
    _caption(fig,
             "Same cells, same arithmetic as 22d, grouped two ways. A cell is "
             "(agent, model, problem) with at least 5 rounds, 148 of them, and every "
             "cell feeds both panels.\n"
             "Swapping the model moves recovery from 0 % to 91 %. Swapping the harness "
             "moves it 11 pp, from kimi at 33 % to swe-mini at 43 %, and the four agents "
             "ran the same nine models, so that band is what harness choice is worth.\n"
             "claude pro is grey and outside the agent span: it is one model on four "
             "cells, the same 20 Opus rounds as the Claude Opus 4.8 dot above, so its "
             "0 % is the model showing through.",
             axes_bottom_in=0.72)
    save(fig, "22f_recovery_model_vs_agent.png")


def mhi_pp(tab, keys):
    """Span of recovery share across keys, in percentage points."""
    v = [tab[k][3] for k in keys]
    return f"{max(v) - min(v):.0f}"


# --------------------------------------------------------------------------- #
# 30 -- pass@1 across the agent x model grid
# --------------------------------------------------------------------------- #
# 22f asked what drives recoverability and answered "the model". This asks the
# same of pass@1 and the answer needs the whole grid, not two spans.
#
# The four real agents ran the same nine models on the same four problems with
# the same round count per problem inside every (agent, model) cell, so the
# problem axis is orthogonal and pooling over it loses nothing.
#
# The margins are EQUAL-WEIGHT means, each model counted once. The alternative,
# sum(passed)/sum(rounds), weights each model by the rounds that agent happened
# to run on it, and the agents ran different totals (claude 340, kimi 220,
# pi 220, swe-mini 320) with different mixes. That is the Simpson mechanism and
# it is live here: pooled, claude is 125/340 = 36.8 % against kimi's
# 86/220 = 39.1 %; equal-weight, claude leads 37.8 % to 36.4 %. Same 560 rounds,
# opposite winner. With one shared weight vector the difference of the agent
# means equals the mean of the nine within-model differences exactly, so no
# reversal is possible.


def _agent_model_grid():
    """(agents, models, cells) over the balanced grid the four real agents share.

    cells[(agent, model)] = (passed, rounds) pooled over the four problems.
    Agents running a single model are dropped, which is what removes claude pro
    (Opus only), and the model list is the intersection, which removes Kimi-K3
    (kimi only). What is left has no holes.
    """
    per = collections.defaultdict(lambda: [0, 0])
    for r in ROWS_LATEST:
        c = per[(display_agent(r), r["model"], r["problem"])]
        c[1] += 1
        c[0] += r["passed"]

    ran = collections.defaultdict(set)
    for a, m, _ in per:
        ran[a].add(m)
    agents = sorted(a for a in ran if len(ran[a]) > 1)
    models = sorted(set.intersection(*(ran[a] for a in agents)))
    problems = sorted({p for _, _, p in per})

    cells = {}
    for a in agents:
        for m in models:
            n = [per[(a, m, p)][1] for p in problems]
            assert min(n) == max(n) > 0, (a, m, n)   # balance is the whole premise
            cells[(a, m)] = (sum(per[(a, m, p)][0] for p in problems), sum(n))
    return agents, models, cells


def _grid_variance(agents, models, rate):
    """Two-way sum-of-squares shares for the pooled grid, plus the noise floor."""
    v = [rate[(a, m)] for a in agents for m in models]
    gm = sum(v) / len(v)
    sst = sum((x - gm) ** 2 for x in v)
    ma = {a: st.mean(rate[(a, m)] for m in models) for a in agents}
    mm = {m: st.mean(rate[(a, m)] for a in agents) for m in models}
    ssa = len(models) * sum((x - gm) ** 2 for x in ma.values())
    ssm = len(agents) * sum((x - gm) ** 2 for x in mm.values())
    return ma, mm, 100 * ssm / sst, 100 * ssa / sst, 100 * (sst - ssm - ssa) / sst


def fig_pass1_agent_model_grid():
    agents, models, cells = _agent_model_grid()
    rate = {k: 100 * c / n for k, (c, n) in cells.items()}
    ma, mm, sh_m, sh_a, sh_i = _grid_variance(agents, models, rate)

    models = sorted(models, key=lambda m: -mm[m])
    agents = sorted(agents, key=lambda a: -ma[a])
    nm, na = len(models), len(agents)

    # One imshow over a (models+1) x (agents+1) array: the extra row and column
    # are the equal-weight margins, on the same colour scale as the cells so the
    # eye can compare them. The corner is masked out.
    grid = np.full((nm + 1, na + 1), np.nan)
    for i, m in enumerate(models):
        for j, a in enumerate(agents):
            grid[i, j] = rate[(a, m)]
        grid[i, na] = mm[m]
    for j, a in enumerate(agents):
        grid[nm, j] = ma[a]

    # Low end is a tint rather than white, so the 0 % cells still read as cells
    # against the page.
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "deck_blue", ["#f1f4f7", "#d7e5f1", BLUE, "#14405f"])
    fig, ax = plt.subplots(figsize=(10.4, 6.6))
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=100,
              aspect="auto", interpolation="nearest")

    for i in range(nm + 1):
        for j in range(na + 1):
            v = grid[i, j]
            if v != v:
                continue
            margin = i == nm or j == na
            col = "white" if v >= 58 else INK
            ax.text(j, i, f"{v:.0f}%" if not margin else f"{v:.1f}%",
                    ha="center", va="center", color=col,
                    fontsize=12 if margin else 13,
                    fontweight="bold" if margin else "normal")

    # Cell borders, and a heavier white gutter in front of the margins.
    for x in np.arange(-0.5, na + 1):
        ax.axvline(x, color="white", linewidth=1.6)
    for y in np.arange(-0.5, nm + 1):
        ax.axhline(y, color="white", linewidth=1.6)
    ax.axvline(na - 0.5, color="white", linewidth=6)
    ax.axhline(nm - 0.5, color="white", linewidth=6)

    ax.set_xticks(range(na + 1), agents + ["model mean"], fontsize=11)
    ax.set_yticks(range(nm + 1), [_mlabel(m) for m in models] + ["agent mean"],
                  fontsize=10)
    for lab in (ax.get_xticklabels()[-1], ax.get_yticklabels()[-1]):
        lab.set_fontweight("bold")
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(visible=False)
    ax.set_ylim(nm + 0.55, -0.5)
    fig.subplots_adjust(bottom=0.03, top=0.88)

    ax.set_title("Pass rate is set by the model. The harness decides whether a "
                 "pairing works at all.", fontsize=14, fontweight="bold", pad=34)
    _caption(fig, axes_bottom_in=0,
             text="pass@1 per (agent, model), pooled over the four problems. "
             "The four agents ran the same nine "
             f"models on the same four problems with the same rounds per problem in "
             f"every cell, {sum(n for _, n in cells.values())} rounds and no holes, so "
             "the problem axis cancels out of this comparison.\n"
             f"Splitting the variance across these {nm * na} cells: the model accounts "
             f"for {sh_m:.0f} %, the agent for {sh_a:.0f} %, and the agent-model "
             f"interaction for {sh_i:.0f} %, of which about 9 points is binomial noise "
             "at these sample sizes. Read as spans, the model moves pass@1 by "
             f"{max(mm.values()) - min(mm.values()):.0f} pp and the agent by "
             f"{max(ma.values()) - min(ma.values()):.0f} pp. Rows separate, columns do "
             "not: six of the nine models vary more across the four agents than the "
             "agent means vary across the nine models, so a row says more than a column "
             "mean does.\n"
             "Margins are equal-weight means, each model counted once. Summing raw "
             "counts instead would weight each model by the rounds that agent happened "
             "to run on it, and the mixes differ, which reverses claude and kimi "
             "(pooled 36.8 % against 39.1 %, equal-weight 37.8 % against 36.4 %). Under "
             "one shared weight vector the gap between two agent means equals the mean "
             "of their nine within-model gaps, so no such reversal can occur.\n"
             "The agent means are too close to rank the agents on. What the agent does "
             "control is whether a pairing runs at all: kimi on Kimi-K2.7-Code is 0 of "
             "20, every round quitting on turn 2 with no solution file, while swe-mini "
             "reaches 98 % on the same model and problems.")
    save(fig, "30_pass1_agent_model_grid.png")


if __name__ == "__main__":
    fig_pass1_by_agent_ci()
    fig_pass1_by_model_ci()
    fig_retry_gain_by_k()
    fig_recovery_by_model()
    fig_recovery_vs_pass1()
    fig_recovery_model_vs_agent()
    fig_failure_by_end_reason()
    fig_failure_owner_by_model()
    fig_adjusted_by_agent()
    fig_adjusted_by_model()
    fig_problem_correlation()
    fig_pass_vs_tokens_model()
    fig_tokens_passed_vs_failed_model()
    fig_pass_vs_dollars_model()
    fig_expected_token_cost_model()
    fig_expected_dollar_cost_model()
    fig_pass1_agent_model_grid()
