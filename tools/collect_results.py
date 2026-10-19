#!/usr/bin/env python3
"""Aggregate per-round pass verdicts into pass@1 per model.

Reads the ``pass_result.json`` files written by ``check_pass.py`` (one per round)
plus the sibling ``facts.json`` for model/problem identity and ``stats.json``
for run cost and how the agent's session ended, and emits:

- a flat JSONL of every run (``--out``)
- a summary with pass@1 per model and a failure breakdown (``--summary``)

Every failing row carries a ``fail_class``: ONE fine-grained slug for why that
round failed, from ``checks/failure_class.py`` — ``context_overflow``,
``format_error``, ``compile_error``, ``mse``, and so on. It replaces the old
answer to that question, the name of the first failed stage, which was the
useless word "build" for nearly half of all rounds (the agent had died before a
compiler ever ran). ``stage_failures`` is still reported next to it so older
readers of summary.json keep working.

Pass/fail itself is unchanged: anything that is not a pass counts as a failure
and every pass@1 is exactly the number it was before.

Run ``check_pass.py`` and ``trace_stats.py`` first so the verdicts and stats
exist. Pure stdlib, host-side.
"""

import argparse
import json
import os
import statistics
from collections import Counter
from glob import glob

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checks"))  # failure_class lives next to the harness
import failure_class as fc


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", default="traces", help="Trace root directory")
    parser.add_argument("--out", default="results/runs.jsonl", help="Output JSONL file")
    parser.add_argument("--summary", default="results/summary.json", help="Summary JSON file")
    return parser.parse_args()


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


# Run-stats fields (from trace_stats.py's stats.json) carried into each run
# record and aggregated in the summary. Kept as a list so both sites stay in
# sync if the set changes.
STAT_FIELDS = ("total_input", "total_output", "total_tokens", "turns", "steps",
               "peak_context", "compacted", "end_reason", "end_severity",
               "capped_calls", "capped_ctx", "empty_reply_nudges",
               "est_total_input", "est_total_output", "est_total_tokens",
               "est_tokenizer_source")

# Model calls per round under the unified limits. claude's --max-turns does not
# count its empty-reply nudges or background-task wakeups, so ``turns`` can
# pass it; those rounds are counted, not dropped.
CALL_LIMIT = 120


def _sum_known(values):
    """Sum of the non-None values, None when every value is None (unknown)."""
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def normalize_model(model):
    """Strip serving prefixes so one model is one row.

    The same weights reach the traces under several ids depending on how the
    gateway/agent was configured — ``Qwen/Qwen3.6-27B``, ``qwen27/Qwen/Qwen3.6-27B``,
    ``local/moonshotai/Kimi-K2.7-Code``, ``deepseek46/DeepSeek-V4-Flash``. Only
    the final segment is the model; the rest is routing. Without this, one
    model's rounds split across several rows and every rate is wrong.

    Two more spellings of the same model: ``Qwen_Qwen3.6-27B`` (remeasured
    rounds take the id from the trace dir, where ``/`` became ``_``, and no
    model name contains ``_``) and ``Qwen3.6-27B-think`` (same weights behind
    the LiteLLM deployment that turns reasoning on, used by claude only).
    """
    name = str(model).rstrip("/").split("/")[-1].split("_")[-1]
    return name.removesuffix("-think") or "unknown"


def _problem_of(traces_root, eval_dir):
    """Problem name = first path segment under the trace root (facts.json has none).

    Trace layout: traces/<PROBLEM>/<AGENT>/<MODEL>/<DATE>/round-N/.eval/
    """
    rel = os.path.relpath(eval_dir, traces_root)
    return rel.split(os.sep)[0]


def _gpu_of_round(eval_dir):
    """Device name from this round's ``gpu_info.txt`` (``GPU 0: NVIDIA A40 (UUID…)``).

    "unknown" when absent: a round that failed before the measure phase (no
    solution.json, build failure) never sampled a GPU.
    """
    try:
        with open(os.path.join(eval_dir, "gpu_info.txt"), "r", encoding="utf-8") as fh:
            line = fh.readline()
    except OSError:
        return "unknown"
    name = line.split(":", 1)[1] if ":" in line else line
    return name.split("(")[0].strip() or "unknown"


def _gpu_of(eval_dir, cache):
    """Device for a round, resolved at the CAMPAIGN level.

    The report filters by device (A40 vs GTX 1050 numbers are not comparable).
    Filtering on the round's own gpu_info.txt would silently drop every FAILING
    round — they have no gpu_info at all — which inflates pass@1 towards 1.0. So
    a round with no gpu_info inherits the device from its sibling rounds in the
    same campaign dir (traces/<problem>/<agent>/<model>/<DATE>/), all of which
    ran on one node.
    """
    own = _gpu_of_round(eval_dir)
    if own != "unknown":
        return own

    campaign = os.path.dirname(os.path.dirname(eval_dir))  # …/<DATE>
    if campaign not in cache:
        found = "unknown"
        for sibling in glob(os.path.join(campaign, "*", ".eval")):
            gpu = _gpu_of_round(sibling)
            if gpu != "unknown":
                found = gpu
                break
        if found == "unknown":
            found = _gpu_by_problem(campaign) or _gpu_by_date(campaign)
        cache[campaign] = found
    return cache[campaign]


def _gpu_by_problem(campaign):
    """Fall back to a device seen elsewhere under the same problem, IF unanimous.

    Needed when EVERY round in a campaign failed before the measure phase, so
    no gpu_info.txt exists anywhere in it — e.g. a campaign that hit a context
    limit on every round. A problem that is genuinely CPU-only never has a
    resolved GPU anywhere under traces/<problem>/, which the date-based
    fallback below cannot express (it assumes everything after 2026-06-22 is
    the A40 and would silently mislabel a CPU node as A40).

    Requiring unanimity across every resolved round in the problem (not just
    one sibling) is deliberate: a problem like RabbitCT-LolaOMP has campaigns
    spanning both the GTX 1050 and A40 eras, so "some sibling resolved to X"
    is not evidence for a campaign whose own era is unknown — only "every
    sibling that ever resolved anything agrees" is. That keeps this function a
    no-op for every existing GPU problem and only fires for a problem type
    that has no GPU at all.
    """
    problem_dir = os.path.dirname(os.path.dirname(os.path.dirname(campaign)))
    resolved = set()
    for sibling in glob(os.path.join(problem_dir, "*", "*", "*", "round-*", ".eval")):
        gpu = _gpu_of_round(sibling)
        if gpu != "unknown":
            resolved.add(gpu)
    if len(resolved) == 1:
        return next(iter(resolved))
    return None


# The two environments never overlapped in time: the GTX 1050 (WSL2) runs
# ended before the A40 (SLURM) campaigns began on 2026-06-22.
A40_ERA_START = "20260622"


def _gpu_by_date(campaign_dir):
    """Last-resort device attribution from the campaign's timestamp dir name.

    Needed for campaigns where NO round reached the measure phase — the pi/Kimi
    and pi/gemma client-config artifacts, where every round died before a GPU
    was ever sampled. Without this their 0/N rows would drop out of the A40
    tables entirely, turning a real 0% into a missing row.
    """
    stamp = os.path.basename(campaign_dir)
    if len(stamp) >= 8 and stamp[:8].isdigit():
        return "NVIDIA A40" if stamp[:8] >= A40_ERA_START else "NVIDIA GeForce GTX 1050"
    return "unknown"


def load_runs(traces_root):
    """One record per round: verdict + identity (facts.json) + run stats (stats.json)."""
    pattern = os.path.join(traces_root, "**", ".eval", "pass_result.json")
    runs = []
    gpu_cache = {}
    for path in glob(pattern, recursive=True):
        verdict = _load_json(path)
        if not verdict:
            continue

        eval_dir = os.path.dirname(path)
        # Rounds cut off by the harness (e.g. endpoint lost mid-solve) are
        # re-run by scripts/phase6/worker.py and must not count.
        if os.path.exists(os.path.join(os.path.dirname(eval_dir), ".round_lost")):
            continue
        facts =_load_json(os.path.join(eval_dir, "facts.json"))
        # stats.json is optional (written by trace_stats.py); absent for old
        # traces or when the agent trace file wasn't captured.
        stats = _load_json(os.path.join(eval_dir, "stats.json"))

        record = {
            "problem": _problem_of(traces_root, eval_dir),
            "gpu": _gpu_of(eval_dir, gpu_cache),
            # Campaign = one scheduled sweep (…/<DATE>/round-N/.eval). Carried so
            # the report can drop single-round smoke/bring-up campaigns.
            "campaign": os.path.dirname(os.path.dirname(eval_dir)),
            "agent": facts.get("agent", "unknown"),
            "model": normalize_model(facts.get("model", "unknown")),
            "model_raw": facts.get("model", "unknown"),
            "round": facts.get("round"),
            "passed": bool(verdict.get("passed")),
            "stages": verdict.get("stages", []),
            "__path": path,
        }
        for field in STAT_FIELDS:
            record[field] = stats.get(field)
        # ONE reason per failing round, combining how the agent's session ended
        # (stats.json) with what the harness found afterwards (facts.json,
        # build.log, run.log). None on a pass.
        klass, detail = fc.fail_class(record["passed"], record["stages"],
                                      record.get("end_reason"), eval_dir, facts)
        record["fail_class"] = klass
        record["fail_detail"] = (detail or "")[:300] or None
        runs.append(record)
    return runs


def failed_stage_names(run):
    """Stages that actually failed (excludes skipped)."""
    return [
        s.get("name")
        for s in run.get("stages", [])
        if not s.get("passed") and not s.get("skipped")
    ]


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _median(values):
    values = sorted(v for v in values if v is not None)
    n = len(values)
    if not n:
        return None
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2


def _num(x):
    """Return x if it's a real number (not bool/None/dict), else None."""
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _stage(run, name):
    for s in run.get("stages", []):
        if s.get("name") == name:
            return s
    return None


def metric_distributions(runs):
    """Continuous metric distributions across a group of runs.

    Complements the binary pass@1 with the underlying numbers so a near-miss is
    visible instead of collapsing to 0/1. Pulled from each stage's ``value``
    (already captured by the checks): mse (float), gpu (``{avg,peak,samples}``),
    perf (throughput fraction of cuBLAS). Perf's value doubles as a bench-MSE on
    its early-fail paths, so only ratios (detail mentions cuBLAS) are counted.
    """
    mses, gpu_avgs, gpu_peaks, perf_fracs = [], [], [], []
    for r in runs:
        s = _stage(r, "mse")
        if s is not None:
            v = _num(s.get("value"))
            if v is not None:
                mses.append(v)
        s = _stage(r, "gpu")
        if s is not None and isinstance(s.get("value"), dict):
            gpu_avgs.append(_num(s["value"].get("avg")))
            gpu_peaks.append(_num(s["value"].get("peak")))
        s = _stage(r, "perf")
        if s is not None and "cuBLAS" in (s.get("detail") or ""):
            v = _num(s.get("value"))
            if v is not None:
                perf_fracs.append(v)

    gpu_peaks = [p for p in gpu_peaks if p is not None]
    return {
        "mse_median": _median(mses),
        "mse_max": max(mses) if mses else None,
        "mse_n": len(mses),
        "gpu_avg_mean": _mean(gpu_avgs),
        "gpu_peak_mean": _mean(gpu_peaks),
        "gpu_peak_min": min(gpu_peaks) if gpu_peaks else None,
        "perf_frac_median": _median(perf_fracs),
        "perf_frac_mean": _mean(perf_fracs),
        "perf_frac_n": len(perf_fracs),
    }


def _mean_std(values):
    """``(mean, std, n)`` over the non-None values; std is the sample stdev.

    n < 2 has no defined spread — std is None there rather than a misleading 0.
    """
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    mean = sum(values) / len(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return mean, std, len(values)


def limit_counts(runs):
    """Limit hits over a set of rounds: capped calls (None = unknown), rounds
    over CALL_LIMIT calls, rounds that hung after the final reply, and claude's
    empty-reply nudges (None for other agents)."""
    return {
        "capped_calls": _sum_known([r.get("capped_calls") for r in runs]),
        "over_call_limit": sum(1 for r in runs if (r.get("turns") or 0) > CALL_LIMIT),
        "hung": sum(1 for r in runs if r.get("end_reason") == "hung_after_final"),
        "empty_reply_nudges": _sum_known([r.get("empty_reply_nudges") for r in runs]),
    }


def per_problem_rankings(runs):
    """Per-(problem, agent, model) ranking rows: pass@1 plus the scalar metrics.

    The continuous metrics are averaged over EVERY round that produced a number,
    pass or fail — a round that built, ran, and then missed the correctness bar
    still carries a meaningful MSE, and dropping it would report 0 for a model
    that is in fact wrong half the time. Rounds that never got that far (build or
    run failed, so no measurement exists) contribute nothing but still count in
    the pass@1 denominator. ``metric_n`` therefore says how many rounds the
    average rests on: ``metric_n < runs`` means the rest crashed before measuring.
    """
    by_problem = {}
    for run in runs:
        key = (run.get("agent", "unknown"), run["model"])
        entry = by_problem.setdefault(run["problem"], {}).setdefault(
            key, {"runs": 0, "passed": 0, "mses": [], "metrics": [], "limit_runs": []}
        )
        entry["runs"] += 1
        entry["limit_runs"].append(run)
        if run["passed"]:
            entry["passed"] += 1

        stage = _stage(run, "mse")
        if stage is not None:
            entry["mses"].append(_num(stage.get("value")))
        stage = _stage(run, "metric")
        if stage is not None and isinstance(stage.get("value"), dict):
            entry["metrics"].append(stage["value"])

    out = {}
    for problem, by_key in by_problem.items():
        rows = []
        metric_name = metric_unit = None
        higher_better = True
        for (agent, model), e in by_key.items():
            values = []
            for m in e["metrics"]:
                # The metric's identity comes from the check itself (metric_stage
                # payload), so the aggregator stays problem-agnostic.
                metric_name = metric_name or m.get("metric")
                metric_unit = metric_unit or m.get("unit")
                if m.get("higher_better") is not None:
                    higher_better = bool(m["higher_better"])
                v = _num(m.get("value"))
                if v is not None:
                    values.append(v)

            mse_mean, mse_std, mse_n = _mean_std(e["mses"])
            met_mean, met_std, met_n = _mean_std(values)
            rows.append({
                "agent": agent,
                "model": model,
                "runs": e["runs"],
                "passed": e["passed"],
                "pass@1": e["passed"] / e["runs"] if e["runs"] else None,
                "mse_mean": mse_mean,
                "mse_std": mse_std,
                "mse_n": mse_n,
                "metric_mean": met_mean,
                "metric_std": met_std,
                "metric_min": min(values) if values else None,
                "metric_max": max(values) if values else None,
                "metric_n": met_n,
                "limits": limit_counts(e["limit_runs"]),
            })
        out[problem] = {
            "metric": metric_name,
            "unit": metric_unit,
            "higher_better": higher_better,
            "rows": rows,
        }
    return out


def _sorted_tally(counts):
    """Counts as a dict ordered biggest-first — the reading order in a report."""
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def summarize(runs):
    # Key by (agent, model) so the same model run by different agents (pi,
    # swe-mini, ...) is reported separately rather than silently merged.
    by_key = {}
    for run in runs:
        key = (run.get("agent", "unknown"), run["model"])
        entry = by_key.setdefault(
            key, {"runs": 0, "passed": 0, "stage_failures": {},
                  "fail_classes": {}, "_runs": []}
        )
        entry["runs"] += 1
        entry["_runs"].append(run)
        if run["passed"]:
            entry["passed"] += 1
        else:
            for name in failed_stage_names(run):
                entry["stage_failures"][name] = entry["stage_failures"].get(name, 0) + 1
            klass = run.get("fail_class") or "unknown"
            entry["fail_classes"][klass] = entry["fail_classes"].get(klass, 0) + 1

    # One corpus-wide failure breakdown, so "why do rounds fail here?" has an
    # answer that does not require summing 40 per-model tables by hand.
    overall = {}
    for run in runs:
        if run["passed"]:
            continue
        klass = run.get("fail_class") or "unknown"
        overall[klass] = overall.get(klass, 0) + 1

    summary = {
        "fail_classes": _sorted_tally(overall),
        "fail_class_labels": {k: fc.FAIL_CLASS_LABEL.get(k, k) for k in overall},
        "agents": {},
        "problems": per_problem_rankings(runs),
    }
    for (agent, model), stats in by_key.items():
        rs = stats["_runs"]
        pass_at_1 = stats["passed"] / stats["runs"] if stats["runs"] else None
        compacted = sum(1 for r in rs if r.get("compacted"))
        summary["agents"].setdefault(agent, {})[model] = {
            "runs": stats["runs"],
            "passed": stats["passed"],
            "pass@1": pass_at_1,
            # why the failures failed, finest first; stage_failures is the old
            # coarse view, kept so existing report scripts still read.
            "fail_classes": _sorted_tally(stats["fail_classes"]),
            "stage_failures": stats["stage_failures"],
            # Continuous metric distributions alongside the binary pass@1.
            "metrics": metric_distributions(rs),
            # Mean run cost/effort across the rounds (None when no stats.json).
            "avg_total_tokens": _mean([r.get("total_tokens") for r in rs]),
            "avg_turns": _mean([r.get("turns") for r in rs]),
            "avg_steps": _mean([r.get("steps") for r in rs]),
            "avg_peak_context": _mean([r.get("peak_context") for r in rs]),
            "compacted_runs": compacted,
            # how every round ended, passes included (hung_after_final and
            # output_limit can end a passing round)
            "end_reasons": _sorted_tally(Counter(r.get("end_reason") or "unknown" for r in rs)),
            "capped_calls": _sum_known([r.get("capped_calls") for r in rs]),
            "over_call_limit_runs": sum(1 for r in rs if (r.get("turns") or 0) > CALL_LIMIT),
            "empty_reply_nudges": _sum_known([r.get("empty_reply_nudges") for r in rs]),
        }
    return summary


def main():
    args = parse_args()
    runs = load_runs(args.traces)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run) + "\n")

    summary = summarize(runs)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
