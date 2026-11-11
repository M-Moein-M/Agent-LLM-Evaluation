#!/usr/bin/env python3
"""Regenerate every figure in results/figures/ from results/runs.jsonl.

    python reference-results/analysis/make_figures.py

Deterministic: same runs.jsonl -> same PNGs. Every figure draws from the same
population: A40 only, every campaign to date, latest campaign per
(problem, agent, model) triple (a handful of explicit multi-campaign pools
and per-round artifact exclusions applied on top, model identities
canonicalized) — see load_latest_all() for the exact, hardcoded rules. So
every number on a slide can be traced back to a trace directory, and reruns
or bug fixes show up on the next regenerate instead of being frozen out by a
stale date window. Only A40 rounds are in scope.

Needs matplotlib + numpy. On the cluster:
    uv venv /tmp/slidesenv --python 3.12
    uv pip install --python /tmp/slidesenv/bin/python matplotlib numpy
    /tmp/slidesenv/bin/python reference-results/analysis/make_figures.py
"""

import collections
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.path.join(REPO, "checks"))
sys.path.insert(0, os.path.join(REPO, "tools"))  # collect_results
from report_tables import EXCLUDE_CAMPAIGNS, _campaign_key  # noqa: E402
from collect_results import normalize_model  # noqa: E402
from failure_class import FAIL_CLASS_LABEL  # noqa: E402

OUT = os.environ.get("EVAL_FIGURES_OUT", os.path.join(REPO, "results", "figures"))
os.makedirs(OUT, exist_ok=True)

# Several plots jitter overlapping points; fix the seed so reruns are identical.
np.random.seed(0)

# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

INK = "#22303f"
BLUE = "#2f6fa8"
GREEN = "#3c8a5b"
RED = "#c0504d"
AMBER = "#d99323"
GREY = "#9aa5b1"
PURPLE = "#6b5b95"


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path, REPO))


def barlabels(ax, bars, fmt="{:.0f}", dx=0, dy=2, **kw):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()),
                    (b.get_x() + b.get_width() / 2 + dx, b.get_height()),
                    ha="center", va="bottom", xytext=(0, dy),
                    textcoords="offset points", fontsize=10, color=INK, **kw)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def stage(r, name):
    for s in r["stages"]:
        if s["name"] == name:
            return s
    return None


def ok(r, name):
    s = stage(r, name)
    return bool(s and s["passed"])


def tok_value(r):
    """Real total_tokens if the provider reported one; otherwise the est_*
    reconstruction (currently populated only for kimi rows, which never get
    metered usage at all — see AGENT.md 'kimi token counts are always null').
    Returns (tokens, estimated_bool)."""
    t = r.get("total_tokens")
    if t:
        return t, False
    e = r.get("est_total_tokens")
    if e:
        return e, True
    return 0, False


def display_agent(r):
    """claude vs claude-pro split for 16a/16b only: same harness binary, but
    Opus (claude-opus-4-8) runs against the direct Anthropic API on a
    subscription token, not the gateway to a self-hosted model — worth
    separating here to compare closed-source vs open-source token cost.
    Every other figure keeps them merged as "claude" (see the merge comment
    in load_latest_all, which is still correct for the harness/pass-rate
    figures — this is a display-only split, ROWS_LATEST itself is untouched)."""
    if r["agent"] == "claude" and r["model"] == "claude-opus-4-8":
        return "claude pro"
    return r["agent"]


DROP_MODELS = {"Kimi-K3"}


def load_latest_all():
    """The single row population every figure in this file draws from — every
    campaign run to date. Rules requested explicitly for this chart:

    - Round-count floor of 3 (not 5) drops single-round debug pokes while admitting
      Kimi-K3's genuine 3-round-per-problem campaigns; checked site-wide that no
      other 3-4-round campaign exists that this would wrongly let in.
    - Latest-only: campaigns are deduped per (problem, agent, model) triple, keeping
      only the most recently dated one. Earlier campaigns for the same pairing are
      superseded, not pooled with the latest.
    - Model-identity merge: pi's self-hosted gemma endpoint sometimes reports the
      generic tag ("gemmaselfhosted/google/gemma-4-31B-it") while actually serving
      FP8-block-quantized weights — same model as local/RedHatAI/...-FP8-block.
      Canonicalized to one key so latest-date selection picks whichever pi campaign
      (bare-tag or FP8-block-tag) is most recent, rather than hiding one under a bug.
    - Artifact exclusion: campaigns confirmed (by inspecting agent.jsonl/facts.json/
      changed_diff.txt directly, not just pass-rate) to be config/harness bugs rather
      than genuine capability failures are dropped as smoke tests.
    """
    raw = [json.loads(l) for l in open(os.environ.get("EVAL_RUNS_JSONL", os.path.join(REPO, "results/runs.jsonl")))]
    rows = [r for r in raw if "A40" in r.get("gpu", "")]
    rows = [r for r in rows if _campaign_key(r) not in EXCLUDE_CAMPAIGNS]

    # A round the harness watchdog killed (.solve_timeout / .measure_timeout)
    # never gets a .eval/pass_result.json, so collect_results.py never emits a
    # row for it at all -- it's invisible to runs.jsonl, not just "excluded".
    # Left alone that silently shrinks a cell's round count (e.g. 18/20
    # instead of 20/20) instead of counting the round as a failure.
    campaign_rounds = collections.defaultdict(set)
    campaign_template = {}
    for r in rows:
        campaign_rounds[r["campaign"]].add(r["round"])
        campaign_template[r["campaign"]] = r
    for campaign, present in campaign_rounds.items():
        camp_dir = os.path.join(REPO, campaign)
        try:
            round_dirs = sorted(d for d in os.listdir(camp_dir) if d.startswith("round-"))
        except OSError:
            continue
        for d in round_dirs:
            n = int(d.split("-")[1])
            if n in present:
                continue
            rd = os.path.join(camp_dir, d)
            if os.path.exists(os.path.join(rd, ".round_lost")):
                continue  # Phase 6 worker re-ran it under a new round number
            if os.path.exists(os.path.join(rd, ".solve_timeout")) or \
               os.path.exists(os.path.join(rd, ".measure_timeout")):
                tmpl = campaign_template[campaign]
                rows.append(dict(tmpl, round=n, passed=False, stages=[], __path=None))

    camps = collections.defaultdict(list)
    for r in rows:
        parts = r["campaign"].rstrip("/").split("/")
        problem, agent, model_dir, date = parts[-4], parts[-3], parts[-2], parts[-1]
        # "claude-pro" is not a different harness: it is the SAME Claude Code
        # binary and the same prompt/tooling, pointed at api.anthropic.com with a
        # subscription OAuth token instead of the gateway. It only lives in a
        # separate agents/apptainer/claude-pro/ tree because the token has to be
        # mounted and the gateway preflight skipped. What differs is the MODEL
        # (claude-opus-4-8 / -5), which is already its own row here, so keeping a
        # separate column would double-count the harness axis.
        if agent == "claude-pro":
            agent = "claude"
        camps[(problem, agent, model_dir, date)].append(r)

    qualifying = []
    for (problem, agent, model_dir, date), rs in camps.items():
        if len(rs) < 3:
            continue
        model = normalize_model(rs[0]["model"])
        # The self-hosted sglang box (10.0.0.1:30000, "gemmasglang"/
        # "gemmaselfhosted" provider in models.local.json) serves FP8-block
        # weights but reports the generic id "google/gemma-4-31B-it" — same
        # underlying model as local/RedHatAI/...-FP8-block. Every agent that
        # hits this endpoint (pi, claude, kimi — confirmed via their launch
        # scripts, e.g. scripts/_run_merged_claude.slurm, scripts/_run_wave_
        # kimi_gemma.slurm) reports the bare tag, so canonicalize it site-wide,
        # not just for pi, so all three land in one row/key.
        if model == "gemma-4-31B-it":
            model = "gemma-4-31B-it-FP8-block"
        # "...-0731-think" is not a second model: it is the same DeepSeek-V4-Flash
        # -0731 weights behind a second LiteLLM deployment that turns reasoning on
        # (merge_reasoning_content_in_choices + chat_template_kwargs.thinking).
        # Only the claude harness used it, because the plain id returns bare text
        # the Anthropic surface then drops. Left un-canonicalized it splits one
        # campaign into two rows, each with a single filled cell.
        if model == "DeepSeek-V4-Flash-0731-think":
            model = "DeepSeek-V4-Flash-0731"
        # NVFP4 is a quantization of Kimi-K2.7-Code, not a separate model, and it
        # only ever ran on two harnesses -- so it folds into the Kimi-K2.7-Code
        # row rather than carrying a row of its own with two cells. The
        # latest-date rule then decides per (problem, agent): pi takes its
        # unquantized unified-limits 20260929 campaigns (28/40, 10 rounds per
        # problem) because they are newer than the NVFP4 sweeps and the earlier
        # unquantized campaigns, while kimi, which never ran unquantized, contributes its
        # NVFP4 campaign (20260805-003151, 1/20).
        #
        # READ THAT 1/20 AS A HARNESS RESULT, NOT A MODEL RESULT. Verified per
        # round in agent.stderr.log: 12 of the 20 rounds aborted with
        # "provider.api_error: The API returned a response containing only
        # thinking content without any text or tool calls", and the provider
        # details printed on that same line read finishReason=completed,
        # rawFinishReason=stop -- i.e. the model returned a COMPLETE response
        # whose text all arrived in reasoning_content. Kimi Code rejects a turn
        # with no content and no tool_calls and kills the run instead of
        # retrying; same class of bug as the claude thinking-block failure that
        # the -think alias works around. A further 7 rounds died on
        # provider.connection_error (timeouts against the self-hosted NVFP4
        # endpoint), leaving exactly 1 round that ran to completion. Per-round
        # exclusion is NOT applied here: it would report 1/1 = 100%.
        if model == "Kimi-K2.7-Code-NVFP4":
            model = "Kimi-K2.7-Code"
        # Kimi-K3 is left out of every figure (user, 2026-10-11): one agent,
        # 3 rounds per problem, run before the unified limits, not re-runnable.
        if model in DROP_MODELS:
            continue
        qualifying.append(dict(problem=problem, agent=agent, model_dir=model_dir,
                                model=model, date=date, rows=rs))

    # (agent, model) -> set of campaign dates to pool instead of picking only
    # the latest. claude+DeepSeek-V4-Flash-0731 ran twice through the -think
    # deployment (08-11 and 08-15); neither run supersedes the other, so both
    # are combined into one 40-round entry per problem rather than dropping
    # the earlier one under the normal latest-date rule.
    MERGE_CAMPAIGNS = {
        ("claude", "DeepSeek-V4-Flash-0731"): {"20260811-084821", "20260815-133051"},
        # swe-mini+Qwen3.6-35B-A3B-FP8: 20260810-082931 was the original 5-
        # rounds/problem campaign; 20260816-115857 was a deliberate deficit-fill
        # (+5 rounds/problem) to match claude's 10 rounds/problem, landed in a
        # fresh dir because the harness always starts numbering at round-1
        # rather than continuing. runtime.memory.txt:335-337 confirms these were
        # meant to be pooled ("10 rounds/problem, joining claude10 -> pooled
        # 40/problem"), not have the later date supersede the earlier one.
        ("swe-mini", "Qwen3.6-35B-A3B-FP8"): {"20260810-082931", "20260816-115857"},
        # pi+Qwen3.6-35B-A3B-FP8: same deficit-fill pattern as swe-mini above --
        # 20260731-193229 (qwenselfhosted provider tag) was the original 5-
        # rounds/problem run, 20260816-103214 (qwen35bselfhosted tag, new
        # endpoint) was the +5 top-up to reach 10 rounds/problem. Pooled rather
        # than latest-only.
        ("pi", "Qwen3.6-35B-A3B-FP8"): {"20260731-193229", "20260816-103214"},
        # kimi+Qwen3.6-35B-A3B-FP8: same deficit-fill pattern -- 20260810-082933
        # was the original 5-rounds/problem run, 20260815-230556 the +5 top-up
        # to reach 10 rounds/problem. Pooled rather than latest-only.
        ("kimi", "Qwen3.6-35B-A3B-FP8"): {"20260810-082933", "20260815-230556"},
        # Phase 10 top-up (2026-10-07): +5 rounds/problem in a new campaign per
        # problem, pooled with the 5-rounds/problem campaign the cell had before
        # (which did not breach the unified limits), to reach 10 rounds/problem.
        ("claude", "Qwen3.8-27B"): {"20260817-185943", "20261007-164859", "20261007-165415",
                                    "20261007-165430", "20261007-165445"},
        ("swe-mini", "Qwen3.8-27B"): {"20260818-051152", "20261007-171000", "20261007-172757",
                                      "20261007-172923", "20261007-173103"},
        ("swe-mini", "DeepSeek-V4-Flash-0731"): {"20260815-171036", "20261007-170318", "20261007-170343",
                                                 "20261007-170404", "20261007-170421"},
        ("pi", "DeepSeek-V4-Flash"): {"20260815-131641", "20261008-174410", "20261008-174540",
                                      "20261008-174555", "20261008-174610"},
    }

    # Pool the merged campaigns first, dated by their newest member, so a later
    # campaign (e.g. a Phase 6 re-run) supersedes the pool whatever the order.
    pools, candidates = {}, []
    for c in qualifying:
        key = (c["problem"], c["agent"], c["model"])
        merge_dates = MERGE_CAMPAIGNS.get((c["agent"], c["model"]))
        if merge_dates and c["date"] in merge_dates:
            if key in pools:
                pools[key]["rows"] = pools[key]["rows"] + c["rows"]
                pools[key]["date"] = max(pools[key]["date"], c["date"])
            else:
                pools[key] = dict(c, rows=list(c["rows"]), merged=True)
        else:
            candidates.append(c)
    latest = {}
    for c in candidates + list(pools.values()):
        key = (c["problem"], c["agent"], c["model"])
        if key not in latest or c["date"] > latest[key]["date"]:
            latest[key] = c

    # Confirmed harness/config bugs, verified per-round by reading each round's
    # agent.jsonl (pi) directly for a stopReason of "length" (response truncated
    # by max_tokens mid-turn, so it never reached the write step) or "error" with
    # an errorMessage showing an infra failure (timeout, context-window overflow).
    # A round left with failure_reason "No solution.json produced" and a clean
    # stopReason of "stop"/"toolUse" is a genuine capability failure and is kept.
    def artifact(c):
        # pi+Kimi-K2.7-Code's broken 20260705/20260706 campaigns (contextWindow >
        # model cap: litellm 400 BadRequestError, 0/40 rounds across all three)
        # are now superseded by the self-hosted unified-limits 20260929 campaigns
        # (28/40, no compaction, server window 256,000) via the
        # latest-date selection above — same situation as the claude+gemma case
        # below, so the blanket exclusion that used to sit here is gone. Without
        # this change the working campaign would be dropped along with the bug.
        # claude+gemma-4-31B-it-FP8-block's broken 20260701 local/RedHatAI campaign
        # (36/40 rounds leaked native Gemma tool-call syntax as text) is naturally
        # superseded by the clean, later self-hosted-sglang campaign (0/40 rounds
        # show the bug, verified across all 4 problems) via latest-date selection
        # above, now that both share the canonicalized model key — no blanket
        # exclusion needed here anymore.
        return False

    # Per-round exclusions: (problem, agent, model) -> set of 1-indexed round
    # numbers confirmed via agent.jsonl stopReason to be infra failures, not
    # capability failures. Every other failing round for these campaigns ended
    # cleanly ("stop") with no infra evidence and is counted.
    #
    # stopReason "length" is NOT excluded (policy set 2026-08-10): a round that
    # spends its whole token budget without producing solution.json is a
    # capability failure, the same event that is already counted as a fail for
    # kimi/swe-mini/claude (which hit their step caps and get no exclusion). The
    # pi+Qwen3.6-35B-A3B-FP8 rounds that used to be dropped here on that basis
    # (LolaBunny {2}, stream-compact {2,3,4,5}, tiled-gemm {1,2,3,4,5}) are now
    # counted, taking that cell from 4/10 to 4/20. Only failures where the
    # endpoint itself did not answer (timeouts below) remain excluded.
    ARTIFACT_ROUNDS = {
        # self-hosted endpoint request timeouts (stopReason "error", "Request timed out.")
        ("RabbitCT-LolaBunny", "pi", "Kimi-K2.7-Code-NVFP4"): {3, 4, 5, 9},
        ("RabbitCT-LolaOMP", "pi", "Kimi-K2.7-Code-NVFP4"): {3, 4},
        ("stream-compact", "pi", "Kimi-K2.7-Code-NVFP4"): {2, 3, 4, 7, 8},
        ("tiled-gemm", "pi", "Kimi-K2.7-Code-NVFP4"): {1, 3, 4},
    }

    out = []
    for c in latest.values():
        if artifact(c):
            continue
        bad_rounds = ARTIFACT_ROUNDS.get((c["problem"], c["agent"], c["model"]), set())
        for r in c["rows"]:
            if r.get("round") in bad_rounds:
                continue
            # project canonicalized model (e.g. gemma merge) and agent
            # (claude-pro -> claude) onto each row
            r = dict(r, model=c["model"], agent=c["agent"])
            out.append(r)
    return out


ROWS_LATEST = load_latest_all()

MODEL_ORDER = ["claude-opus-4-8", "DeepSeek-V4-Flash", "Qwen3.6-27B", "Kimi-K2.7-Code",
               "gemma-4-31B-it-FP8-block", "gpt-oss-120b", "Llama-3.3-70B-Instruct"]
MODEL_SHORT = {
    "claude-opus-4-8": "Claude Opus 4.8\n(frontier, API)",
    "DeepSeek-V4-Flash": "DeepSeek-V4-Flash",
    "Qwen3.6-27B": "Qwen3.6-27B",
    "Kimi-K2.7-Code": "Kimi-K2.7-Code",
    "gemma-4-31B-it-FP8-block": "gemma-4-31B-it",
    "gpt-oss-120b": "gpt-oss-120b",
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B",
}
# Figures 05 and 05b only: row order by model family, and labels without the
# bracketed suffixes. MODEL_ORDER / MODEL_SHORT stay as they are for the others.
FIG05_MODEL_ORDER = ["claude-opus-4-8",
                     "DeepSeek-V4-Flash", "DeepSeek-V4-Flash-0731",
                     "Qwen3.6-27B", "Qwen3.6-35B-A3B-FP8", "Qwen3.8-27B",
                     "Kimi-K2.7-Code",
                     "gemma-4-31B-it-FP8-block", "gpt-oss-120b", "Llama-3.3-70B-Instruct"]
FIG05_LABEL = {"claude-opus-4-8": "Claude Opus 4.8"}
AGENT_ORDER = ["claude-pro", "swe-mini", "claude", "pi", "kimi"]
PROBLEMS = ["RabbitCT-LolaBunny", "RabbitCT-LolaOMP", "stream-compact", "tiled-gemm"]
PROB_SHORT = {"RabbitCT-LolaBunny": "LolaBunny\nreal CT\nbackprojection",
              "RabbitCT-LolaOMP": "LolaOMP\nreal CT,\nOpenMP variant",
              "stream-compact": "stream-compact\nplanted\nrace bug",
              "tiled-gemm": "tiled-gemm\nplanted launch-\nconfig bug"}


# --------------------------------------------------------------------------- #
# fail-class clusters, feed 23a in make_figures_stats.py
# --------------------------------------------------------------------------- #
# Root-cause groups for the 20-odd fail_class slugs -- fewer bars than the
# raw vocabulary, split by *pipeline stage* (where the harness caught it),
# not by guessed intent. Every fail_class in FAIL_CLASS_LABEL must appear
# here; anything not explicitly placed falls into "Infra hiccup".
FAIL_CLUSTER = {
    "mse": "Wrong output", "digest": "Wrong output",
    "metric": "Wrong output", "integrity": "Wrong output",
    "perf": "Underperforms", "gpu": "Underperforms",
    "run_error": "Runtime errors", "run_timeout": "Runtime errors",
    "compile_error": "Software defect (build failed)",
    "link_error": "Software defect (build failed)",
    "toolchain_missing": "Software defect (build failed)",
    "make_config_stop": "Software defect (build failed)",
    "make_no_rule": "Software defect (build failed)",
    "build_error": "Software defect (build failed)",
    "ptxas_error": "Software defect (build failed)",
    "missing_file": "Software defect (build failed)",
    "build_timeout": "Software defect (build failed)",
    "no_solution": "Agent never finished",
    "incomplete_solution": "Agent never finished",
    "context_overflow": "Agent never finished",
    "max_turns": "Agent never finished",
    "max_steps_exceeded": "Agent never finished",
    "output_limit": "Agent never finished",
    "cut_off": "Agent never finished",
    "limits_exceeded": "Agent never finished",
    "thinking_block": "Agent never finished",
}

# A couple of representative slugs per cluster, for the bar label -- not
# the full membership (see FAIL_CLUSTER above), and never `no_solution`
# for "Agent never finished": its name just restates the cluster name.
FAIL_CLUSTER_EXAMPLE = {
    "Wrong output": "MSE above the per-problem cap (1e-9 to 1e-2)",
    "Underperforms": "too slow vs reference (perf), GPU idle (gpu)",
    "Runtime errors": "graded run crashed (run_error)",
    "Software defect (build failed)": "compile error, config.mk never re-run",
    "Agent never finished": "context window exceeded, turn cap reached",
}


# --------------------------------------------------------------------------- #
# 05 — pass@1 matrix
# --------------------------------------------------------------------------- #
def fig_pass_matrix():
    # Uses ROWS_LATEST (all-time, latest-campaign-per-pairing, artifacts excluded)
    # -- the same population every figure in this file draws from.
    cnt = collections.defaultdict(lambda: [0, 0])
    for r in ROWS_LATEST:
        c = cnt[(r["agent"], r["model"])]
        c[1] += 1
        c[0] += r["passed"]
    agents = [a for a in AGENT_ORDER if any(k[0] == a for k in cnt)]
    seen_models = {m for _, m in cnt}
    models = [m for m in FIG05_MODEL_ORDER if m in seen_models]
    models += sorted(seen_models - set(models))

    def label(m):
        return FIG05_LABEL.get(m) or MODEL_SHORT.get(m, m)

    grid = np.full((len(models), len(agents)), np.nan)
    txt = {}
    for (a, m), (p, n) in cnt.items():
        i, j = models.index(m), agents.index(a)
        grid[i, j] = 100 * p / n
        txt[(i, j)] = f"{p}/{n}\n{100*p/n:.0f}%"

    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    ax.imshow(np.where(np.isnan(grid), -20, grid), cmap="RdYlGn", vmin=-20, vmax=100, aspect="auto")
    for (i, j), t in txt.items():
        ax.text(j, i, t, ha="center", va="center", fontsize=10, color=INK, fontweight="bold")
    for i in range(len(models)):
        for j in range(len(agents)):
            if np.isnan(grid[i, j]):
                # Blank white cell, no label: an empty cell already reads as
                # "not run" and the text competed with the filled cells.
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="white", edgecolor="white"))
    ax.set_xticks(range(len(agents)), agents)
    ax.set_yticks(range(len(models)), [label(m).replace("\n", " ") for m in models])
    ax.set_title("pass@1 by model × harness")
    ax.set_xlabel("agent harness")
    ax.grid(False)
    save(fig, "05_pass_matrix.png")


# --------------------------------------------------------------------------- #
# 05b — pass@1 matrix, percentage only
# --------------------------------------------------------------------------- #
def fig_pass_matrix_pct():
    # Same cells, same colour scale as 05_pass_matrix.png. Only the "p/n" line is
    # dropped, so the grid reads as a heatmap at slide distance. The round counts
    # stay in 05 for anyone who needs them.
    cnt = collections.defaultdict(lambda: [0, 0])
    for r in ROWS_LATEST:
        c = cnt[(r["agent"], r["model"])]
        c[1] += 1
        c[0] += r["passed"]
    agents = [a for a in AGENT_ORDER if any(k[0] == a for k in cnt)]
    seen_models = {m for _, m in cnt}
    models = [m for m in FIG05_MODEL_ORDER if m in seen_models]
    models += sorted(seen_models - set(models))

    def label(m):
        return FIG05_LABEL.get(m) or MODEL_SHORT.get(m, m)

    grid = np.full((len(models), len(agents)), np.nan)
    txt = {}
    for (a, m), (p, n) in cnt.items():
        i, j = models.index(m), agents.index(a)
        grid[i, j] = 100 * p / n
        txt[(i, j)] = f"{100*p/n:.0f}%"

    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    ax.imshow(np.where(np.isnan(grid), -20, grid), cmap="RdYlGn", vmin=-20, vmax=100, aspect="auto")
    for (i, j), t in txt.items():
        ax.text(j, i, t, ha="center", va="center", fontsize=13, color=INK, fontweight="bold")
    for i in range(len(models)):
        for j in range(len(agents)):
            if np.isnan(grid[i, j]):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="white", edgecolor="white"))
    ax.set_xticks(range(len(agents)), agents)
    ax.set_yticks(range(len(models)), [label(m).replace("\n", " ") for m in models])
    ax.set_title("pass@1 by model × harness")
    ax.set_xlabel("agent harness")
    ax.grid(False)
    save(fig, "05b_pass_matrix_pct.png")


# --------------------------------------------------------------------------- #
# 06 — problem difficulty
# --------------------------------------------------------------------------- #
def fig_problem_difficulty():
    cnt = collections.defaultdict(lambda: [0, 0])
    for r in ROWS_LATEST:
        c = cnt[r["problem"]]
        c[1] += 1
        c[0] += r["passed"]
    vals = [100 * cnt[p][0] / cnt[p][1] for p in PROBLEMS]
    ns = [f"{cnt[p][0]}/{cnt[p][1]}" for p in PROBLEMS]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.bar([PROB_SHORT[p] for p in PROBLEMS], vals,
                  color=[RED, AMBER, GREEN, BLUE], width=0.55)
    for b, lbl in zip(bars, ns):
        ax.annotate(f"{b.get_height():.0f}%\n({lbl})",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points", fontsize=11, color=INK)
    ax.set_ylabel("pass@1 (%)")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title("pass@1 by problem")
    save(fig, "06_problem_difficulty.png")


# --------------------------------------------------------------------------- #
# 10a — total token spend, by pairing
# --------------------------------------------------------------------------- #
def fig_tokens_total():
    label_override = {"gemma-4-31B-it-FP8-block": "gemma-4-31B-it\n(FP8-block)"}

    def label(m):
        return label_override.get(m) or MODEL_SHORT.get(m, m)

    g = collections.defaultdict(list)
    won = collections.defaultdict(int)
    est = collections.defaultdict(bool)
    for r in ROWS_LATEST:
        tok, estimated = tok_value(r)
        if tok > 0:
            g[(r["agent"], r["model"])].append(tok)
            won[(r["agent"], r["model"])] += r["passed"]
            est[(r["agent"], r["model"])] = est[(r["agent"], r["model"])] or estimated
    keys = sorted(g, key=lambda k: sum(g[k]))
    vals = [sum(g[k]) for k in keys]
    labs = [f"{k[0]} + {label(k[1]).splitlines()[0]}" for k in keys]

    fig, ax = plt.subplots(figsize=(9.4, 0.4 * len(keys) + 2))
    bars = ax.barh(labs, vals, color=[GREEN if won[k] else RED for k in keys], height=0.62)
    for b, k in zip(bars, keys):
        if est[k]:
            b.set_hatch("////")
        ax.text(b.get_width() * 1.03, b.get_y() + b.get_height() / 2,
                f"{b.get_width()/1e6:,.1f} M  (n={len(g[k])})", va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("total prompt+completion tokens spent, all rounds to date (log scale)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREEN),
                       plt.Rectangle((0, 0), 1, 1, color=RED),
                       plt.Rectangle((0, 0), 1, 1, facecolor=GREY, hatch="////")],
              labels=["solved at least one problem", "never solved anything",
                      "kimi: estimated, not metered"],
              frameon=False, fontsize=10, loc="lower right")
    ax.set_title("Total token spend by (agent, model)")
    save(fig, "10a_total_tokens_by_pairing.png")


# --------------------------------------------------------------------------- #
# 10b — average tokens per passed round, by pairing
# --------------------------------------------------------------------------- #
def fig_tokens_avg_passed():
    label_override = {"gemma-4-31B-it-FP8-block": "gemma-4-31B-it\n(FP8-block)"}

    def label(m):
        return label_override.get(m) or MODEL_SHORT.get(m, m)

    g = collections.defaultdict(list)
    est = collections.defaultdict(bool)
    for r in ROWS_LATEST:
        if not r["passed"]:
            continue
        tok, estimated = tok_value(r)
        if tok > 0:
            g[(r["agent"], r["model"])].append(tok)
            est[(r["agent"], r["model"])] = est[(r["agent"], r["model"])] or estimated
    keys = sorted(g, key=lambda k: sum(g[k]) / len(g[k]))
    vals = [sum(g[k]) / len(g[k]) for k in keys]
    labs = [f"{k[0]} + {label(k[1]).splitlines()[0]}" for k in keys]

    fig, ax = plt.subplots(figsize=(9.4, 0.4 * len(keys) + 2))
    bars = ax.barh(labs, vals, color=BLUE, height=0.62)
    for b, k in zip(bars, keys):
        if est[k]:
            b.set_hatch("////")
        ax.text(b.get_width() * 1.03, b.get_y() + b.get_height() / 2,
                f"{b.get_width()/1e3:,.0f} k  (n={len(g[k])})", va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("mean prompt+completion tokens per passed round (log scale)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=BLUE),
                       plt.Rectangle((0, 0), 1, 1, facecolor=GREY, hatch="////")],
              labels=["metered", "kimi: estimated, not metered"],
              frameon=False, fontsize=10, loc="lower right")
    ax.set_title("Average token cost per passed round, by (agent, model)")
    save(fig, "10b_avg_tokens_per_passed_round.png")


# --------------------------------------------------------------------------- #
# 12 — tiled-gemm performance
# --------------------------------------------------------------------------- #
def fig_gemm_perf():
    # Uses ROWS_LATEST (all-time, latest-campaign-per-pairing, artifacts excluded),
    # so this chart reflects every pairing that shows up in 05_pass_matrix.png.
    g = collections.defaultdict(list)
    for r in ROWS_LATEST:
        if r["problem"] != "tiled-gemm":
            continue
        s = stage(r, "metric")
        if s and isinstance(s.get("value"), dict) and isinstance(s["value"].get("value"), (int, float)):
            g[(r["agent"], r["model"])].append((s["value"]["value"], ok(r, "mse")))
    keys = sorted(g, key=lambda k: st.mean(v for v, _ in g[k]))

    def label(m):
        return MODEL_SHORT.get(m, m).splitlines()[0]

    labs = [f"{k[0]} + {label(k[1])}" for k in keys]
    xmax = max(x for v in g.values() for x, _ in v)

    fig, ax = plt.subplots(figsize=(9.6, 0.32 * len(keys) + 2.2))
    bad_labeled = False
    for i, k in enumerate(keys):
        v = g[k]
        vals = [x for x, _ in v]
        good = [x for x, passed in v if passed]
        bad = [x for x, passed in v if not passed]
        yg = np.random.normal(i, 0.07, len(good))
        yb = np.random.normal(i, 0.07, len(bad))
        ax.scatter(good, yg, s=42, color=BLUE, alpha=0.7, zorder=3)
        ax.scatter(bad, yb, s=42, color=RED, alpha=0.85, zorder=3,
                   label=None if (not bad or bad_labeled) else "failed MSE check (wrong answer)")
        bad_labeled = bad_labeled or bool(bad)
        ax.scatter([st.mean(vals)], [i], s=130, marker="|", color=INK, zorder=4, linewidths=2.5)
    ax.axvline(1.0, color=GREEN, ls="-", lw=1.6)
    ax.axvline(0.126, color=RED, ls="--", lw=1.6)
    ax.axvline(0.15, color=AMBER, ls=":", lw=1.6)
    top = len(keys) - 0.35
    ax.text(1.01, top, "cuBLAS parity", color=GREEN, fontsize=10, va="top")
    ax.text(0.118, top, "untouched starter kernel (12.6 %)", color=RED, fontsize=10,
            va="top", ha="right")
    ax.text(0.163, top - 0.75, "pass floor 15 %", color=AMBER, fontsize=10, va="top")
    ax.set_yticks(range(len(keys)), labs)
    ax.set_ylim(-0.7, len(keys) - 0.1)
    ax.set_xlabel("kernel speed, as a multiple of cuBLAS (× cuBLAS; paired cudaEvent timing)")
    ax.set_xlim(-0.02, xmax * 1.06)
    handles, labels_ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels_, loc="lower right", frameon=False, fontsize=9)
    ax.set_title("tiled-gemm: every dot is one round, all runs to date.\n"
                 "Red = fast but wrong (failed the MSE check)")
    save(fig, "12_gemm_perf.png")


# --------------------------------------------------------------------------- #
# 12b/12c — RabbitCT reconstruction error (MSE) variance, by agent / by model
# --------------------------------------------------------------------------- #
def _rabbitct_mse_by(key_fn):
    # LolaBunny and LolaOMP are pooled: both check against the same
    # Reference256.vol at the same 1e-6 MSE bar, and check.py documents the
    # OMP port as required to be "numerically equivalent" to the Bunny
    # baseline -- same measurement, two implementation routes, not two
    # different metrics. ROWS_LATEST, so this reflects 05_pass_matrix.png.
    #
    # Passed + failed-but-measured, not passed-only: among passed rounds MSE is
    # exactly 0.0 (CT reconstruction here is exact-or-broken), so a passed-only
    # CV is undefined. Failing rounds' nonzero MSE is what carries the variance
    # signal, so this reads as "how consistent is this agent/model's
    # correctness", not a repeatability-of-a-passing-solution measure.
    #
    # Not every round in 05_pass_matrix.png: a round that failed at build/run
    # never produced a run.log MSE line, so it has no metric stage and is
    # invisible here (345 of 566 pooled RabbitCT rounds carry a value). That
    # tilts these charts in favour of agents that fail EARLY (build) over ones
    # that fail LATE (wrong numbers) -- same n < runs caveat results/tables.md
    # prints under each table.
    g = collections.defaultdict(list)
    for r in ROWS_LATEST:
        if r["problem"] not in ("RabbitCT-LolaBunny", "RabbitCT-LolaOMP"):
            continue
        s = stage(r, "metric")
        if s and isinstance(s.get("value"), dict) and isinstance(s["value"].get("value"), (int, float)):
            g[key_fn(r)].append(s["value"]["value"])
    keys = [k for k in g if len(g[k]) > 2 and st.mean(g[k]) > 0]
    cv = {k: st.stdev(g[k]) / st.mean(g[k]) for k in keys}
    keys = sorted(keys, key=lambda k: cv[k], reverse=True)  # lowest CV on top
    return keys, cv, g


def fig_rabbitct_mse_by_agent():
    keys, cv, g = _rabbitct_mse_by(lambda r: r["agent"])
    vals = [cv[k] for k in keys]

    fig, ax = plt.subplots(figsize=(9.6, 0.4 * len(keys) + 1.5))
    bars = ax.barh(keys, vals, color=BLUE, height=0.55)
    for b in bars:
        ax.text(b.get_width() + 0.1, b.get_y() + b.get_height() / 2,
                f"{b.get_width():.2f}", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, max(vals) * 1.2)
    ax.set_xlabel("coefficient of variation of RabbitCT MSE, passed + failed-but-measured rounds")
    ax.set_title("RabbitCT correctness variance, by agent\n(LolaBunny + LolaOMP pooled)")
    save(fig, "12b_rabbitct_mse_by_agent.png")


def fig_rabbitct_mse_by_model():
    label_override = {"gemma-4-31B-it-FP8-block": "gemma-4-31B-it\n(FP8-block)"}
    keys, cv, g = _rabbitct_mse_by(lambda r: r["model"])
    vals = [cv[k] for k in keys]
    labs = [_model_label(k, label_override) for k in keys]

    fig, ax = plt.subplots(figsize=(9.6, 0.4 * len(keys) + 1.2))
    bars = ax.barh(labs, vals, color=BLUE, height=0.55)
    for b in bars:
        ax.text(b.get_width() + 0.1, b.get_y() + b.get_height() / 2,
                f"{b.get_width():.2f}", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, max(vals) * 1.2)
    ax.set_xlabel("coefficient of variation of RabbitCT MSE, passed + failed-but-measured rounds")
    ax.set_title("RabbitCT correctness variance, by model\n(LolaBunny + LolaOMP pooled)")
    save(fig, "12c_rabbitct_mse_by_model.png")


# --------------------------------------------------------------------------- #
# 12e — RabbitCT reconstruction error (MSE) ranking, by model
#
# BY-MODEL ONLY, deliberately. The matching by-agent chart (12d) was built and
# then dropped: mean MSE per agent measures gemma-4-31B-it exposure, not agent
# quality. Every one of the worst rounds for claude/pi/swe-mini is a gemma
# round (max MSE 294 / 1263 / 3324), while kimi's worst round anywhere is 2.84
# -- and kimi drew only 10 gemma rounds to the others' 20, because its gemma
# campaign is the pinned pre-EXCLUDE_CAMPAIGNS one. Macro-averaged over the
# five non-gemma models all four agents ran, the agents sit at 0.0034-0.0060
# (claude 2.85, from its Qwen3.6-35B-A3B-FP8 cell) and kimi is third, not
# first. gemma-at-219 is a real MODEL property, so this chart is sound; the
# agent split was not. See AGENT.md "Aggregating a metric across models".
# --------------------------------------------------------------------------- #
def _rabbitct_mse_mean_by(key_fn):
    # Same pooling and same round population as _rabbitct_mse_by (LolaBunny +
    # LolaOMP, passed + failed-but-measured).
    # Mean, not median: with most pairings passing over half their rounds,
    # median collapses to exactly 0 for nearly everyone and stops
    # discriminating. Mean stays informative because rare-but-severe
    # failures (MSE up to ~3300 HU^2) pull it up in proportion to how often
    # and how badly a pairing gets it wrong -- one number for both failure
    # rate and failure severity.
    g = collections.defaultdict(list)
    for r in ROWS_LATEST:
        if r["problem"] not in ("RabbitCT-LolaBunny", "RabbitCT-LolaOMP"):
            continue
        s = stage(r, "metric")
        if s and isinstance(s.get("value"), dict) and isinstance(s["value"].get("value"), (int, float)):
            g[key_fn(r)].append(s["value"]["value"])
    keys = [k for k in g if len(g[k]) > 2]
    mean = {k: st.mean(g[k]) for k in keys}
    keys = sorted(keys, key=lambda k: mean[k], reverse=True)  # lowest (best) on top
    return keys, mean, g


def fig_rabbitct_mse_mean_by_model():
    label_override = {"gemma-4-31B-it-FP8-block": "gemma-4-31B-it\n(FP8-block)"}
    keys, mean, g = _rabbitct_mse_mean_by(lambda r: r["model"])
    vals = [mean[k] for k in keys]
    labs = [_model_label(k, label_override) for k in keys]

    fig, ax = plt.subplots(figsize=(8.8, 0.4 * len(keys) + 1.2))
    bars = ax.barh(labs, vals, color=BLUE, height=0.55)
    for b in bars:
        ax.annotate(f"{b.get_width():.3g}", xy=(b.get_width(), b.get_y() + b.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=10, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(min(vals) / 3, max(vals) * 8)
    ax.set_xlabel("mean RabbitCT MSE, passed + failed-but-measured rounds — log scale")
    ax.set_title("RabbitCT reconstruction error, by model\n(LolaBunny + LolaOMP pooled)")
    save(fig, "12e_rabbitct_mse_mean_by_model.png")


# --------------------------------------------------------------------------- #
# 13 — determinism
# --------------------------------------------------------------------------- #
def fig_determinism():
    g = collections.defaultdict(list)
    for r in ROWS_LATEST:
        if r["problem"] != "tiled-gemm":
            continue
        s = stage(r, "metric")
        if s and isinstance(s.get("value"), dict) and isinstance(s["value"].get("value"), (int, float)):
            g[(r["agent"], r["model"])].append(s["value"]["value"])
    keys = [k for k in g if len(g[k]) > 2 and k[1] != "Llama-3.3-70B-Instruct"]
    cv = {k: (st.stdev(g[k]) / st.mean(g[k])) for k in keys}
    keys = sorted(keys, key=lambda k: cv[k], reverse=True)
    labs = [f"{k[0]} + {MODEL_SHORT.get(k[1], k[1]).splitlines()[0]}" for k in keys]
    vals = [cv[k] for k in keys]

    fig, ax = plt.subplots(figsize=(9.0, 0.32 * len(keys) + 1.5))
    colors = []
    for v in vals:
        if v < 0.15:
            colors.append(GREEN)
        elif v < 0.35:
            colors.append(AMBER)
        else:
            colors.append(RED)
    bars = ax.barh(labs, vals, color=colors, height=0.6)
    for b in bars:
        ax.text(b.get_width() + 0.015, b.get_y() + b.get_height() / 2,
                f"{b.get_width():.2f}", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, max(vals) * 1.12)
    ax.set_xlabel("coefficient of variation of tiled-gemm kernel speed (× cuBLAS) across rounds")
    ax.set_title("Run-to-run spread is a model property")
    save(fig, "13_determinism.png")


# --------------------------------------------------------------------------- #
# 13b — determinism, correctness (MSE) instead of speed
# --------------------------------------------------------------------------- #
def fig_determinism_mse(problem, out_name, label):
    g = collections.defaultdict(list)
    for r in ROWS_LATEST:
        if r["problem"] != problem or not r["passed"]:
            continue
        s = stage(r, "mse")
        if s and isinstance(s.get("value"), (int, float)):
            g[(r["agent"], r["model"])].append(s["value"])
    keys = [k for k in g if len(g[k]) > 2 and k[1] != "Llama-3.3-70B-Instruct"
            and st.mean(g[k]) > 0]
    cv = {k: (st.stdev(g[k]) / st.mean(g[k])) for k in keys}
    keys = sorted(keys, key=lambda k: cv[k], reverse=True)
    labs = [f"{k[0]} + {MODEL_SHORT.get(k[1], k[1]).splitlines()[0]}" for k in keys]
    vals = [cv[k] for k in keys]

    fig, ax = plt.subplots(figsize=(9.0, 0.32 * len(keys) + 1.5))
    colors = [GREEN if v < 0.15 else AMBER if v < 0.35 else RED for v in vals]
    bars = ax.barh(labs, vals, color=colors, height=0.6)
    for b, k in zip(bars, keys):
        ax.text(b.get_width() + 0.015, b.get_y() + b.get_height() / 2,
                f"{b.get_width():.2f} (n={len(g[k])})", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel(f"coefficient of variation of {label} MSE across passed rounds")
    ax.set_title("Run-to-run spread in correctness (MSE) is a model property")
    save(fig, out_name)


# --------------------------------------------------------------------------- #
# 14 — GPU gate: avg vs peak
# --------------------------------------------------------------------------- #
def fig_gpu_gate():
    TH = {"RabbitCT-LolaBunny": (20, 50), "RabbitCT-LolaOMP": (20, 50),
          "stream-compact": (15, 40), "tiled-gemm": (15, 40)}
    pts = []
    for r in ROWS_LATEST:
        s = stage(r, "gpu")
        if s and isinstance(s.get("value"), dict) and s["value"].get("samples"):
            pts.append((r["problem"], s["value"]["avg"], s["value"]["peak"], r["passed"]))
    saved = [p for p in pts if p[1] < TH[p[0]][0] and p[2] >= TH[p[0]][1] and p[3]]

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for lbl, sel, c, m in [("failed round", False, GREY, "o"), ("passed round", True, GREEN, "o")]:
        xs = [p[1] for p in pts if p[3] == sel]
        ys = [p[2] for p in pts if p[3] == sel]
        ax.scatter(xs, ys, s=26, color=c, alpha=0.55, label=lbl, marker=m)
    xs = [p[1] for p in saved]
    ys = [p[2] for p in saved]
    ax.scatter(xs, ys, s=90, facecolor="none", edgecolor=RED, linewidths=1.8,
               label=f"correct, but an average-based gate\nwould have failed them (n={len(saved)})")
    ax.axvline(20, color=AMBER, ls="--", lw=1.2)
    ax.axhline(50, color=BLUE, ls="--", lw=1.2)
    ax.text(20.6, 4, "old avg gate (20 %)", color=AMBER, fontsize=9, rotation=90, va="bottom")
    ax.text(1, 51.5, "peak gate (50 %)", color=BLUE, fontsize=9)
    ax.set_xlabel("mean GPU utilisation over the graded run (%)")
    ax.set_ylabel("peak GPU utilisation (%)")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 105)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Why we gate on peak, not average GPU utilisation")
    save(fig, "14_gpu_gate.png")


# --------------------------------------------------------------------------- #
# 16a — cost per solve, by agent
# --------------------------------------------------------------------------- #
def fig_cost_per_solve_agent():
    # Every other figure merges the "claude-pro" harness (Opus over the direct
    # Anthropic API) into "claude" (see the merge comment in load_latest_all)
    # since the model axis already distinguishes it. Here we deliberately
    # split it back out via display_agent() so this chart can compare
    # closed-source (claude pro / Opus) against open-source (claude, talking
    # to self-hosted models through the gateway) token cost side by side.
    g = collections.defaultdict(lambda: {"tok": 0, "n": 0, "est": False})
    passed_total = collections.defaultdict(int)
    for r in ROWS_LATEST:
        if not r["passed"]:
            continue
        agent = display_agent(r)
        passed_total[agent] += 1
        tok, estimated = tok_value(r)
        if tok <= 0:
            continue
        e = g[agent]
        e["tok"] += tok
        e["n"] += 1
        e["est"] = e["est"] or estimated
    keys = [k for k in g if g[k]["n"] > 0]
    keys = sorted(keys, key=lambda k: g[k]["tok"] / g[k]["n"])
    vals = [g[k]["tok"] / g[k]["n"] / 1e6 for k in keys]
    partial = {k: passed_total[k] - g[k]["n"] for k in keys if passed_total[k] > g[k]["n"]}

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    bars = ax.barh(keys, vals, color=BLUE, height=0.55)
    for b, k in zip(bars, keys):
        if g[k]["est"]:
            b.set_hatch("////")
        ax.text(b.get_width() + max(vals) * 0.02, b.get_y() + b.get_height() / 2,
                f"{b.get_width():.1f} M", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, max(vals) * 1.55)
    ax.set_xlabel("million tokens per problem solved (solved rounds only)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=BLUE)]
    labels = ["metered"]
    if any(g[k]["est"] for k in keys):
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=GREY, hatch="////"))
        labels.append("kimi: estimated, not metered")
    # Below the axes, not inside — inside collides with the bar-end labels no
    # matter which corner (5 bars now leaves no bar-free rectangle wide enough
    # for the longer "claude pro"/"kimi: estimated" labels).
    ax.legend(handles=handles, labels=labels, frameon=False, fontsize=9, ncol=len(handles),
              loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.subplots_adjust(bottom=0.3)
    ax.set_title("Token cost per solve, by agent")
    save(fig, "16a_cost_per_solve_by_agent.png")


# --------------------------------------------------------------------------- #
# 16b — cost per solve, by model
# --------------------------------------------------------------------------- #
def fig_cost_per_solve_model():
    # "claude-opus-4-8" is the only closed-source model here (every other bar
    # is a self-hosted model); color/label it to match 16a's "claude pro"
    # split so the two figures read consistently as closed- vs open-source.
    # label() below keeps only the first line, so "claude pro" must be on the
    # same line as the model name to actually render on the bar.
    label_override = {"gemma-4-31B-it-FP8-block": "gemma-4-31B-it\n(FP8-block)",
                       "claude-opus-4-8": "Claude Opus 4.8 (claude pro)"}

    def label(m):
        return (label_override.get(m) or MODEL_SHORT.get(m, m)).splitlines()[0]

    g = collections.defaultdict(lambda: {"tok": 0, "n": 0, "est": False})
    for r in ROWS_LATEST:
        if not r["passed"]:
            continue
        tok, estimated = tok_value(r)
        if tok <= 0:
            continue
        e = g[r["model"]]
        e["tok"] += tok
        e["n"] += 1
        e["est"] = e["est"] or estimated
    keys = [k for k in g if g[k]["n"] > 0]
    keys = sorted(keys, key=lambda k: g[k]["tok"] / g[k]["n"])
    vals = [g[k]["tok"] / g[k]["n"] / 1e6 for k in keys]
    labs = [label(k) for k in keys]

    fig, ax = plt.subplots(figsize=(8.4, 0.45 * len(keys) + 1.5))
    bars = ax.barh(labs, vals, color=BLUE, height=0.55)
    for b, k in zip(bars, keys):
        if g[k]["est"]:
            b.set_hatch("////")
        ax.text(b.get_width() + max(vals) * 0.02, b.get_y() + b.get_height() / 2,
                f"{b.get_width():.1f} M", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, max(vals) * 1.3)
    ax.set_xlabel("million tokens per problem solved, pooled across agents (solved rounds only)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=BLUE)]
    labels = ["fully metered"]
    if any(g[k]["est"] for k in keys):
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=GREY, hatch="////"))
        labels.append("includes estimated (kimi) rows")
    ax.legend(handles=handles, labels=labels, frameon=False, fontsize=10, loc="lower right")
    ax.set_title("Token cost per solve, by model")
    save(fig, "16b_cost_per_solve_by_model.png")


# --------------------------------------------------------------------------- #
# 16c / 16d — the same cost-per-solve bars, split into prompt vs completion
# --------------------------------------------------------------------------- #
# 16a/16b answer "how many tokens does one solve cost". They do not say what
# those tokens are. Almost all of the volume is prompt (the whole conversation
# plus every tool result is re-sent on every turn, and for the Anthropic
# surface total_input also folds in cache_read/cache_creation -- see the usage
# accounting in checks/trace_stats.py). Splitting the bar shows that the cost
# is re-reading context, not writing code, which is what makes prompt caching
# and context trimming the levers that matter.
IN_COLOR = "#2f6fa8"      # BLUE, prompt side
OUT_COLOR = "#d99323"     # AMBER, completion side


def io_split(r):
    """(input, output, estimated) for one row, rescaled so input + output
    equals exactly the tok_value() total 16a/16b already plot. Every row but
    one is already internally consistent; rescaling keeps the new bars'
    totals identical to 16a/16b instead of quietly diverging on that row."""
    total, estimated = tok_value(r)
    if total <= 0:
        return 0.0, 0.0, estimated
    if estimated:
        i = r.get("est_total_input") or 0
        o = r.get("est_total_output") or 0
    else:
        i = r.get("total_input") or 0
        o = r.get("total_output") or 0
    s = i + o
    if s <= 0:
        return 0.0, 0.0, estimated
    k = total / s
    return i * k, o * k, estimated


def _tok_fmt(v):
    """Token count as M/K. The M form keeps one decimal so the total here reads
    identically to the same bar's label in 16a/16b."""
    if v >= 1e6:
        return f"{v / 1e6:.1f} M"
    if v >= 1e3:
        return f"{v / 1e3:.0f} K"
    return f"{v:.0f}"


def _cost_split_bars(ax, keys, labs, g, xlabel):
    """Shared drawing for 16c/16d: stacked prompt/completion bars, per-solve
    means, sorted by total (same order as 16a/16b)."""
    # keys arrive sorted ascending; index 0 at the bottom matches how 16a/16b
    # stack their bars (biggest on top), so the two charts read the same way.
    y = np.arange(len(keys))
    tot = np.array([(g[k]["inp"] + g[k]["out"]) / g[k]["n"] / 1e6 for k in keys])
    segs = [("prompt (input)", IN_COLOR, np.array([g[k]["inp"] / g[k]["n"] / 1e6 for k in keys])),
            ("completion (output)", OUT_COLOR, np.array([g[k]["out"] / g[k]["n"] / 1e6 for k in keys]))]

    left = np.zeros(len(keys))
    for name, color, w in segs:
        bars = ax.barh(y, w, left=left, height=0.55, color=color, label=name,
                       edgecolor="white", linewidth=0.6)
        for b, k in zip(bars, keys):
            if g[k]["est"]:
                b.set_hatch("////")
        left += w

    # The completion segment is ~2 % wide, so no in-bar text fits and none is
    # legible over the estimated-rows hatch either. Absolute counts and the
    # share go in the end-of-bar label instead.
    for yy, t, k in zip(y, tot, keys):
        i = g[k]["inp"] / g[k]["n"]
        o = g[k]["out"] / g[k]["n"]
        share_in = 100 * i / (i + o)
        ax.text(t + max(tot) * 0.02, yy,
                f"{_tok_fmt((i + o))}:  {_tok_fmt(i)} ({share_in:.0f}%) in"
                f"  /  {_tok_fmt(o)} out",
                va="center", fontsize=9.5, color=INK)

    ax.set_yticks(y, labs)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlim(0, max(tot) * 2.15)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def _cost_split_legend(ax, g, keys, est_label="kimi: estimated, not metered", **kw):
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=IN_COLOR),
               plt.Rectangle((0, 0), 1, 1, facecolor=OUT_COLOR)]
    labels = ["prompt (input, incl. cached context)", "completion (output)"]
    if any(g[k]["est"] for k in keys):
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=GREY, hatch="////"))
        labels.append(est_label)
    ax.legend(handles=handles, labels=labels, frameon=False, fontsize=9, **kw)


def fig_cost_split_agent():
    # Same population, same "claude pro" split and same sort key as 16a, so
    # the two figures line up bar for bar.
    g = collections.defaultdict(lambda: {"inp": 0.0, "out": 0.0, "n": 0, "est": False})
    for r in ROWS_LATEST:
        if not r["passed"]:
            continue
        i, o, estimated = io_split(r)
        if i + o <= 0:
            continue
        e = g[display_agent(r)]
        e["inp"] += i
        e["out"] += o
        e["n"] += 1
        e["est"] = e["est"] or estimated
    keys = sorted((k for k in g if g[k]["n"] > 0),
                  key=lambda k: (g[k]["inp"] + g[k]["out"]) / g[k]["n"])

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    _cost_split_bars(ax, keys, keys, g,
                     "million tokens per problem solved (solved rounds only)")
    _cost_split_legend(ax, g, keys, ncol=3, loc="upper center",
                       bbox_to_anchor=(0.5, -0.22))
    fig.subplots_adjust(bottom=0.3)
    ax.set_title("What the token cost per solve is made of, by agent")
    save(fig, "16c_cost_split_by_agent.png")


def fig_cost_split_model():
    label_override = {"gemma-4-31B-it-FP8-block": "gemma-4-31B-it\n(FP8-block)",
                      "claude-opus-4-8": "Claude Opus 4.8 (claude pro)"}

    def label(m):
        return (label_override.get(m) or MODEL_SHORT.get(m, m)).splitlines()[0]

    g = collections.defaultdict(lambda: {"inp": 0.0, "out": 0.0, "n": 0, "est": False})
    for r in ROWS_LATEST:
        if not r["passed"]:
            continue
        i, o, estimated = io_split(r)
        if i + o <= 0:
            continue
        e = g[r["model"]]
        e["inp"] += i
        e["out"] += o
        e["n"] += 1
        e["est"] = e["est"] or estimated
    keys = sorted((k for k in g if g[k]["n"] > 0),
                  key=lambda k: (g[k]["inp"] + g[k]["out"]) / g[k]["n"])

    fig, ax = plt.subplots(figsize=(9.2, 0.5 * len(keys) + 2.4))
    _cost_split_bars(ax, keys, [label(k) for k in keys], g,
                     "million tokens per problem solved, pooled across agents (solved rounds only)")
    # Below the axes, not "lower right": the bottom bar is the longest one
    # (sorted ascending) so an in-axes legend lands on top of it.
    # A model bar is hatched when the kimi *agent* contributed rows to it, same
    # wording as 16b -- the bar itself is not a kimi model.
    _cost_split_legend(ax, g, keys, est_label="includes estimated (kimi) rows",
                       ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.subplots_adjust(bottom=0.22)
    ax.set_title("What the token cost per solve is made of, by model")
    save(fig, "16d_cost_split_by_model.png")


# --------------------------------------------------------------------------- #
# 16e — what the peak context is made of, by model
# --------------------------------------------------------------------------- #
# 16c/16d split the BILLED token total, and there the completion side is ~2 %.
# That number is an artifact of the accounting: total_input sums the prompt of
# every model call, and the whole conversation is re-sent each call, so one
# output token gets re-billed as input on every later call (checked per call in
# both pi and swe-mini traces: the jump in prompt size from one call to the
# next always covers the previous call's full output, reasoning included).
# Measured against the conversation ONCE instead — the largest single context
# the round ever assembled — the model's own writing is about two thirds of it.
#
# Only pi and swe-mini can be measured this way. Claude Code's per-message
# output_tokens in agent.stream.jsonl are partial (single-digit counts per
# event; the usable total lives only in the terminal "result" record, which has
# no per-call series), gateway-routed claude rounds report no per-message usage
# at all, and kimi reports none anywhere.
_PEAK_AGENTS = {"pi": "agent.jsonl",
                "swe-mini": "agent.traj.json"}
_peak_cache = {}


def _usage_series_pi(p):
    out = []
    with open(p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "message_end":
                continue
            msg = ev.get("message") or {}
            if msg.get("role") != "assistant":
                continue
            u = msg.get("usage") or {}
            out.append((int(u.get("input") or 0), int(u.get("output") or 0),
                        int(u.get("reasoning") or 0)))
    return out


def _usage_series_swemini(p):
    d = json.load(open(p, encoding="utf-8", errors="replace"))
    msgs = d if isinstance(d, list) else (d.get("messages") or d.get("trajectory") or [])
    out = []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        u = (((m.get("extra") or {}).get("response") or {}).get("usage")) or {}
        det = u.get("completion_tokens_details") or {}
        out.append((int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0),
                    int(det.get("reasoning_tokens") or 0)))
    return out


def usage_series(r):
    """Per-model-call [(prompt, completion, reasoning), ...] for one round, or
    None when the round's trace carries no usable per-call usage. Calls with a
    zero prompt are dropped (usage the provider never relayed)."""
    agent = r.get("agent")
    fn = _PEAK_AGENTS.get(agent)
    if not fn or not r.get("__path"):
        return None
    path = os.path.join(REPO, os.path.dirname(os.path.dirname(r["__path"])), fn)
    if path not in _peak_cache:
        try:
            s = (_usage_series_pi if agent == "pi" else _usage_series_swemini)(path)
            s = [c for c in s if c[0] > 0] or None
        except (OSError, ValueError):
            s = None
        _peak_cache[path] = s
    return _peak_cache[path]


def peak_composition(r):
    """(peak_context, tokens the model itself wrote inside it) for one round.

    Finds the call with the largest prompt — that call's prompt IS the peak
    context — and sums the completions of every call before it. Everything left
    over is task prompt, system prompt and tool results."""
    series = usage_series(r)
    if not series:
        return None
    k = max(range(len(series)), key=lambda j: series[j][0])
    peak = series[k][0]
    own = min(sum(c[1] for c in series[:k]), peak)
    return peak, own


def fig_context_composition_model():
    label_override = {"gemma-4-31B-it-FP8-block": "gemma-4-31B-it\n(FP8-block)"}

    def label(m):
        return (label_override.get(m) or MODEL_SHORT.get(m, m)).splitlines()[0]

    g = collections.defaultdict(lambda: {"peak": 0, "own": 0, "n": 0})
    for r in ROWS_LATEST:
        # Compaction rewrites the history mid-run, so output emitted before a
        # compaction point is not necessarily still inside the peak context.
        # Only 4 of the 243 measurable rounds compacted; drop them rather than
        # overstate the model-written share.
        if not r["passed"] or r.get("compacted"):
            continue
        pc = peak_composition(r)
        if not pc:
            continue
        peak, own = pc
        e = g[r["model"]]
        e["peak"] += peak
        e["own"] += own
        e["n"] += 1
    keys = sorted((k for k in g if g[k]["n"] > 0), key=lambda k: g[k]["peak"] / g[k]["n"])
    y = np.arange(len(keys))
    own = np.array([g[k]["own"] / g[k]["n"] / 1e3 for k in keys])
    rest = np.array([(g[k]["peak"] - g[k]["own"]) / g[k]["n"] / 1e3 for k in keys])
    tot = own + rest

    fig, ax = plt.subplots(figsize=(9.2, 0.5 * len(keys) + 2.4))
    ax.barh(y, own, height=0.55, color=OUT_COLOR, edgecolor="white", linewidth=0.6,
            label="written by the model")
    ax.barh(y, rest, left=own, height=0.55, color=GREY, edgecolor="white", linewidth=0.6,
            label="task prompt, system prompt, tool results")
    for yy, o, t in zip(y, own, tot):
        ax.text(t + max(tot) * 0.02, yy,
                f"{t:.0f} K:  {o:.0f} K model-written ({100 * o / t:.0f}%)",
                va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, [label(k) for k in keys])
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlim(0, max(tot) * 1.62)
    ax.set_xlabel("thousand tokens in the largest context assembled "
                  "(pi + swe-mini, solved rounds only)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    fig.subplots_adjust(bottom=0.22)
    ax.set_title("Composition of the peak context, by model")
    save(fig, "16e_context_composition_by_model.png")


# --------------------------------------------------------------------------- #
# 16f — token cost per solve if prior-turn thinking is dropped from history
# --------------------------------------------------------------------------- #
# Today every agent here replays the full assistant message, thinking included,
# on every later call — verified per call in pi and swe-mini traces: the jump
# in prompt size from one call to the next covers the previous call's whole
# completion, reasoning and all. So a reasoning block emitted at call j is
# billed again on each of the (N - 1 - j) calls that follow it.
#
# The counterfactual here is the common alternative: keep the current turn's
# thinking, strip it from every earlier turn. Saving is then exact from the
# per-call series, no modelling:
#
#     saved = sum_j reasoning_j * (N - 1 - j)
#
# It holds the trajectory fixed. A model that no longer sees its own earlier
# reasoning may well act differently and need more or fewer turns; that is not
# measurable from these traces, so this is a pure re-billing question.
_THINK_MIN_ROUNDS = 5


def thinking_resend_saving(r):
    """(total tokens billed, tokens saved by dropping prior-turn thinking) for
    one round. None when the trace reports no reasoning tokens — DeepSeek and
    gemma endpoints never broke reasoning out, so those rounds cannot be
    measured and are not counted as zero."""
    series = usage_series(r)
    if not series or not any(c[2] for c in series):
        return None
    n = len(series)
    billed = sum(c[0] + c[1] for c in series)
    saved = sum(rs * (n - 1 - j) for j, (_, _, rs) in enumerate(series))
    return billed, min(saved, sum(c[0] for c in series))


def fig_thinking_resend_model():
    def label(m):
        return MODEL_SHORT.get(m, m).splitlines()[0]

    g = collections.defaultdict(lambda: {"billed": 0, "saved": 0, "n": 0})
    for r in ROWS_LATEST:
        if not r["passed"]:
            continue
        v = thinking_resend_saving(r)
        if not v:
            continue
        billed, saved = v
        e = g[r["model"]]
        e["billed"] += billed
        e["saved"] += saved
        e["n"] += 1
    keys = sorted((k for k in g if g[k]["n"] >= _THINK_MIN_ROUNDS),
                  key=lambda k: g[k]["billed"] / g[k]["n"])
    y = np.arange(len(keys))
    kept = np.array([(g[k]["billed"] - g[k]["saved"]) / g[k]["n"] / 1e6 for k in keys])
    saved = np.array([g[k]["saved"] / g[k]["n"] / 1e6 for k in keys])
    tot = kept + saved

    fig, ax = plt.subplots(figsize=(9.2, 0.62 * len(keys) + 2.6))
    ax.barh(y, kept, height=0.55, color=BLUE, edgecolor="white", linewidth=0.6,
            label="still billed")
    ax.barh(y, saved, left=kept, height=0.55, color=AMBER, edgecolor="white",
            linewidth=0.6, label="prior-turn thinking, no longer re-sent")
    for yy, kp, t in zip(y, kept, tot):
        ax.text(t + max(tot) * 0.02, yy,
                f"{t:.1f} M \u2192 {kp:.1f} M   ({100 * (t - kp) / t:.0f}% cut)",
                va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, [f"{label(k)}\n(n={g[k]['n']} rounds)" for k in keys], fontsize=9)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlim(0, max(tot) * 1.5)
    ax.set_xlabel("million tokens per problem solved (rounds reporting reasoning tokens)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    fig.subplots_adjust(bottom=0.24)
    ax.set_title("Token cost per solve if prior-turn thinking is dropped from history")
    save(fig, "16f_thinking_resend_by_model.png")





# --------------------------------------------------------------------------- #
# 17a — pass@1, aggregated by agent
# --------------------------------------------------------------------------- #
def fig_pass1_by_agent():
    # Rollup of 05_pass_matrix.png along the agent axis (pooled across every
    # model that agent ran), using the same ROWS_LATEST counts as the matrix
    # so the numbers reconcile. Unlike 16a, claude-pro stays merged into
    # claude here (no display_agent() split): a pooled pass@1 bar built from
    # a single (agent, model) pairing (n=15/20) doesn't fit alongside bars
    # pooling several pairings each — it's not a meaningful "closed vs
    # open-source" comparison at this granularity, just one cell of the
    # matrix relabeled as a bar.
    cnt = collections.defaultdict(lambda: [0, 0])
    for r in ROWS_LATEST:
        c = cnt[r["agent"]]
        c[1] += 1
        c[0] += r["passed"]
    keys = sorted(cnt, key=lambda k: cnt[k][0] / cnt[k][1])
    vals = [100 * cnt[k][0] / cnt[k][1] for k in keys]

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    bars = ax.barh(keys, vals, color=BLUE, height=0.55)
    for b, k in zip(bars, keys):
        p, n = cnt[k]
        ax.text(b.get_width() + 2, b.get_y() + b.get_height() / 2,
                f"{100*p/n:.0f}%", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, 108)
    ax.set_xlabel("pass@1 (%)")
    ax.set_title("pass@1, by agent")
    save(fig, "17a_pass1_by_agent.png")


# --------------------------------------------------------------------------- #
# 17b — pass@1, aggregated by model
# --------------------------------------------------------------------------- #
def fig_pass1_by_model():
    # Rollup of 05_pass_matrix.png along the model axis (pooled across every
    # agent that ran it). No claude-opus-4-8 special-casing here (unlike
    # 16b): only "claude" ever ran it, so its bar is really just one
    # (agent, model) pairing pooled across 4 problems, same as 17a's dropped
    # claude-pro split — not a meaningful highlight, so it's plotted like
    # every other model.
    label_override = {"gemma-4-31B-it-FP8-block": "gemma-4-31B-it\n(FP8-block)"}

    def label(m):
        return (label_override.get(m) or MODEL_SHORT.get(m, m)).splitlines()[0]

    cnt = collections.defaultdict(lambda: [0, 0])
    for r in ROWS_LATEST:
        # Only "claude" ever ran this model, so its bar would be one (agent,
        # model) pairing pooled across 4 problems, not a genuine multi-pairing
        # rollup like every other bar here — dropped entirely rather than
        # shown plain (see 17a for the by-agent side of this call).
        if r["model"] == "claude-opus-4-8":
            continue
        c = cnt[r["model"]]
        c[1] += 1
        c[0] += r["passed"]
    keys = sorted(cnt, key=lambda k: cnt[k][0] / cnt[k][1])
    vals = [100 * cnt[k][0] / cnt[k][1] for k in keys]
    labs = [label(k) for k in keys]

    fig, ax = plt.subplots(figsize=(8.4, 0.45 * len(keys) + 1.2))
    bars = ax.barh(labs, vals, color=BLUE, height=0.55)
    for b, k in zip(bars, keys):
        p, n = cnt[k]
        ax.text(b.get_width() + 2, b.get_y() + b.get_height() / 2,
                f"{100*p/n:.0f}%", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, 108)
    ax.set_xlabel("pass@1 (%)")
    ax.set_title("pass@1, by model")
    save(fig, "17b_pass1_by_model.png")


# --------------------------------------------------------------------------- #
# shared helpers for 18a/18b
# --------------------------------------------------------------------------- #
def _pass_cells():
    """(agent, model) -> [passed, total] from ROWS_LATEST -- the exact same
    cells as 05_pass_matrix.png, so every number below reconciles with the
    matrix cell-for-cell."""
    cnt = collections.defaultdict(lambda: [0, 0])
    for r in ROWS_LATEST:
        c = cnt[(r["agent"], r["model"])]
        c[1] += 1
        c[0] += r["passed"]
    return cnt


# claude-opus-4-8 was only ever run by claude, Kimi-K3 only by kimi -- "best
# agent" / "spread across agents" for those models is undefined (one
# candidate), same exclusion rule 17b already applies when rolling up the
# model axis. Not excluded from the by-agent figure (18a): there, each
# is one real cell of that agent's own row, not a single pairing standing in
# for an average.
_SINGLE_AGENT_MODELS = {"claude-opus-4-8", "Kimi-K3"}


def _model_label(m, label_override=None):
    o = (label_override or {}).get(m) or MODEL_SHORT.get(m, m)
    return o.splitlines()[0]


# --------------------------------------------------------------------------- #
# 18a -- best model per agent
# --------------------------------------------------------------------------- #
def fig_best_model_by_agent():
    # Each agent's single best (agent, model) cell -- unlike 17a's pooled
    # rollup this is one real matrix cell, not several pairings averaged
    # together, so claude-opus-4-8 is a legitimate "best" here (not dropped
    # the way 17a/17b drop it): it really is claude's best result, same
    # framing as slide 4's "frontier anchor" bullet.
    cnt = _pass_cells()
    best = {}
    for (agent, model), (p, n) in cnt.items():
        pct = 100 * p / n
        if agent not in best or pct > best[agent][1]:
            best[agent] = (model, pct)
    keys = sorted(best, key=lambda a: best[a][1])
    vals = [best[a][1] for a in keys]
    labs_txt = [f"{_model_label(best[a][0])}  {best[a][1]:.0f}%" for a in keys]

    fig, ax = plt.subplots(figsize=(8.4, 0.5 * len(keys) + 1.2))
    bars = ax.barh(keys, vals, color=BLUE, height=0.55)
    for b, t in zip(bars, labs_txt):
        ax.text(b.get_width() + 3, b.get_y() + b.get_height() / 2,
                t, va="center", fontsize=10, color=INK)
    ax.set_xlim(0, 165)
    ax.set_xlabel("pass@1 (%) of that agent's best model")
    ax.set_title("Best model per agent")
    save(fig, "18a_best_model_by_agent.png")


# --------------------------------------------------------------------------- #
# 18b -- best agent per model
# --------------------------------------------------------------------------- #
def fig_best_agent_by_model():
    # Mirror of 18a on the model axis. Skips the two single-agent models
    # (see _SINGLE_AGENT_MODELS) -- "best of one" isn't a comparison.
    cnt = _pass_cells()
    best = {}
    for (agent, model), (p, n) in cnt.items():
        if model in _SINGLE_AGENT_MODELS:
            continue
        pct = 100 * p / n
        if model not in best or pct > best[model][1]:
            best[model] = (agent, pct)
    keys = sorted(best, key=lambda m: best[m][1])
    vals = [best[m][1] for m in keys]
    label_override = {"gemma-4-31B-it-FP8-block": "gemma-4-31B-it\n(FP8-block)"}
    labs = [_model_label(m, label_override) for m in keys]

    def bar_text(m):
        agent, pct = best[m]
        # Llama-3.3-70B: all 4 agents score 0% -- a 4-way tie at zero isn't a
        # "winner," so say so instead of naming an arbitrary agent.
        if pct == 0:
            return "0%  — no agent solved it"
        return f"{agent}  {pct:.0f}%"

    fig, ax = plt.subplots(figsize=(8.4, 0.5 * len(keys) + 1.2))
    bars = ax.barh(labs, vals, color=BLUE, height=0.55)
    for b, m in zip(bars, keys):
        ax.text(b.get_width() + 3, b.get_y() + b.get_height() / 2,
                bar_text(m), va="center", fontsize=10, color=INK)
    ax.set_xlim(0, 165)
    ax.set_xlabel("pass@1 (%) of that model's best agent")
    ax.set_title("Best agent per model")
    save(fig, "18b_best_agent_by_model.png")


if __name__ == "__main__":
    fig_pass_matrix()
    fig_pass_matrix_pct()
    fig_problem_difficulty()
    fig_tokens_total()
    fig_tokens_avg_passed()
    fig_gemm_perf()
    fig_rabbitct_mse_by_agent()
    fig_rabbitct_mse_by_model()
    fig_rabbitct_mse_mean_by_model()
    fig_determinism()
    fig_determinism_mse("tiled-gemm", "13b_determinism_mse.png", "tiled-gemm")
    fig_gpu_gate()
    fig_cost_per_solve_agent()
    fig_cost_per_solve_model()
    fig_cost_split_agent()
    fig_cost_split_model()
    fig_context_composition_model()
    fig_thinking_resend_model()
    fig_pass1_by_agent()
    fig_pass1_by_model()
    fig_best_model_by_agent()
    fig_best_agent_by_model()
