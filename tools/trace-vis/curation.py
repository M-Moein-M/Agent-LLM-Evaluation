"""The "important runs" selection — the same set of rounds that
`reference-results/figures/05_pass_matrix.png` is built from.

The viewer defaults to this scope; `scope=all` in the API turns it off. The
rules are a port of `reference-results/analysis/make_figures.py::load_latest_all()` (plus
`EXCLUDE_CAMPAIGNS`/`_campaign_key` from `reference-results/analysis/report_tables.py`), reproduced
here rather than imported because the image must build and selftest with no
workspace bound, and because `make_figures.py` pulls in matplotlib/numpy.
`selftest.py --workspace` diffs the exclusion set against the workspace copy so
the two cannot drift silently.

The rules, in order:

1. A40 rounds only. Anything measured elsewhere (the CPU-only 2D-seq study,
   rounds that died before sampling a GPU) is out of scope for the matrix.
2. Campaigns in EXCLUDE_CAMPAIGNS are dropped as harness artifacts.
3. Campaigns with fewer than 3 rounds are dropped as debug pokes. The floor is
   3, not 5, so Kimi-K3's genuine 3-round-per-problem campaigns are kept.
4. Identity merges — `claude-pro` is the claude harness pointed at
   api.anthropic.com, and three model ids are alternate spellings of a model
   that already has its own identity. Merging them is what makes rule 5 pick
   the newest campaign for a pairing instead of hiding one behind the other.
5. Latest campaign per (problem, agent, model) wins; earlier campaigns for the
   same pairing are superseded, not pooled.
6. Individual rounds confirmed to be infra failures (endpoint timeouts) are
   dropped. Note that a round that merely ran out of tokens is a capability
   failure and is kept — same policy as the figures.

The merges affect *selection only*. The viewer keeps showing each run under the
agent and model directory it actually ran in, so navigation still matches
`traces/`.
"""
import collections
import os

# --- from checks/report_tables.py -------------------------------------------
# claude-pro hit "You've hit your session limit" mid-sweep on these two
# campaigns; each was re-run clean and is superseded by the campaign named
# alongside it.
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
}

MIN_CAMPAIGN_ROUNDS = 3

# claude-pro is the SAME Claude Code binary and prompt as claude, pointed at
# api.anthropic.com with a subscription token instead of the gateway; it
# only has its own agents/ tree because the token has to be mounted. What
# differs is the model, which is its own identity already.
AGENT_ALIASES = {"claude-pro": "claude"}

MODEL_ALIASES = {
    # the self-hosted sglang box serves FP8-block weights but reports the
    # generic id; every agent that hits it reports the bare tag
    "gemma-4-31B-it": "gemma-4-31B-it-FP8-block",
    # same 0731 weights behind a second LiteLLM deployment with reasoning on
    "DeepSeek-V4-Flash-0731-think": "DeepSeek-V4-Flash-0731",
    # a quantization of Kimi-K2.7-Code, not a separate model
    "Kimi-K2.7-Code-NVFP4": "Kimi-K2.7-Code",
}

# Per-round drops: (problem, agent, model) -> 1-indexed rounds where the
# endpoint itself never answered (stopReason "error", "Request timed out."),
# verified in each round's agent.jsonl. stopReason "length" is deliberately NOT
# here: burning the token budget without producing solution.json is a
# capability failure (policy set 2026-08-10).
#
# The model ids here are the ones the traces report, i.e. BEFORE MODEL_ALIASES;
# upstream looks these up with the merged id, so the table is inert there. It is
# matched against both spellings below. That changes nothing for the current
# data — every campaign it names lost the latest-date race to pi's clean
# unquantized 20260809 campaign anyway — but it keeps the table from being a
# silent no-op if an older campaign ever becomes the latest one again.
ARTIFACT_ROUNDS = {
    ("RabbitCT-LolaBunny", "pi", "Kimi-K2.7-Code-NVFP4"): {3, 4, 5, 9},
    ("RabbitCT-LolaOMP", "pi", "Kimi-K2.7-Code-NVFP4"): {3, 4},
    ("stream-compact", "pi", "Kimi-K2.7-Code-NVFP4"): {2, 3, 4, 7, 8},
    ("tiled-gemm", "pi", "Kimi-K2.7-Code-NVFP4"): {1, 3, 4},
}

SCOPES = ("curated", "all")
# "curated" encodes the owner's reference selection (A40 only, hand-picked exclusions) and only
# makes sense for the reference results. Default to every run; TRACE_VIS_SCOPE=curated opts in.
DEFAULT_SCOPE = "curated" if os.environ.get("TRACE_VIS_SCOPE") == "curated" else "all"


def normalize_scope(value):
    v = (value or "").strip().lower()
    if v in ("", "default"):
        return DEFAULT_SCOPE
    if v in ("important", "latest", "curated"):
        return "curated"
    return "all" if v == "all" else DEFAULT_SCOPE


def campaign_key(row):
    """`<problem>/<agent>/<model_dir>/<date>` — the campaign path's 4-segment tail."""
    camp = (row.get("campaign_rel") or "").rstrip("/")
    parts = camp.split("/")
    return "/".join(parts[-4:]) if len(parts) >= 4 else camp


def canonical_model(model):
    return MODEL_ALIASES.get(model, model)


def canonical_agent(agent):
    return AGENT_ALIASES.get(agent, agent)


def _on_a40(row):
    return "A40" in (row.get("gpu") or "")


def curated_run_ids(rows):
    """The run_ids that survive every rule above.

    `rows` are index rows (see indexer._read_runs_jsonl) in runs.jsonl order —
    the order matters only for which row names a campaign's model when a
    campaign directory somehow holds more than one.
    """
    rows = [r for r in rows if r.get("has_eval") and _on_a40(r)]
    rows = [r for r in rows if campaign_key(r) not in EXCLUDE_CAMPAIGNS]

    camps = collections.OrderedDict()
    for r in rows:
        parts = (r.get("campaign_rel") or "").rstrip("/").split("/")
        model_dir = parts[-2] if len(parts) >= 2 else r["model"]
        key = (r["problem"], canonical_agent(r["agent"]), model_dir, r["campaign"])
        camps.setdefault(key, []).append(r)

    latest = {}
    for (problem, agent, _model_dir, date), rs in camps.items():
        if len(rs) < MIN_CAMPAIGN_ROUNDS:
            continue
        raw_model = rs[0]["model"]
        key = (problem, agent, canonical_model(raw_model))
        if key not in latest or date > latest[key][0]:
            latest[key] = (date, raw_model, rs)

    out = set()
    for (problem, agent, model), (_date, raw_model, rs) in latest.items():
        bad = (ARTIFACT_ROUNDS.get((problem, agent, model))
               or ARTIFACT_ROUNDS.get((problem, agent, raw_model), ()))
        for r in rs:
            if r.get("round") in bad:
                continue
            out.add(r["run_id"])
    return out


def workspace_exclude_campaigns(workspace):
    """EXCLUDE_CAMPAIGNS as the workspace itself defines it, or None.

    Read as text instead of imported: reference-results/analysis/report_tables.py is a CLI module
    and the container may not have it bound at all.
    """
    path = os.path.join(workspace, "reference-results", "analysis", "report_tables.py")
    try:
        src = open(path, "r", errors="replace").read()
    except OSError:
        return None
    head = src.split("EXCLUDE_CAMPAIGNS", 1)
    if len(head) < 2 or "{" not in head[1]:
        return None
    body = head[1].split("{", 1)[1].split("}", 1)[0]
    out = set()
    for line in body.splitlines():
        line = line.strip().rstrip(",").strip()
        if line[:1] in ("'", '"'):
            out.add(line.strip("\"'"))
    return out
