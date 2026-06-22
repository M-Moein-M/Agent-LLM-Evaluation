#!/usr/bin/env python3
"""Aggregate per-round pass verdicts into pass@1 per model.

Reads the ``pass_result.json`` files written by ``check_pass.py`` (one per round)
plus the sibling ``facts.json`` for model/problem identity, and emits:

- a flat JSONL of every run (``--out``)
- a summary with pass@1 per model and a per-stage failure breakdown (``--summary``)

Run ``check_pass.py`` first so the verdicts exist. Pure stdlib, host-side.
"""

import argparse
import json
import os
from glob import glob


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
               "peak_context", "compacted")


def load_runs(traces_root):
    """One record per round: verdict + identity (facts.json) + run stats (stats.json)."""
    pattern = os.path.join(traces_root, "**", ".eval", "pass_result.json")
    runs = []
    for path in glob(pattern, recursive=True):
        verdict = _load_json(path)
        if not verdict:
            continue

        eval_dir = os.path.dirname(path)
        facts = _load_json(os.path.join(eval_dir, "facts.json"))
        # stats.json is optional (written by trace_stats.py); absent for old
        # traces or when the agent trace file wasn't captured.
        stats = _load_json(os.path.join(eval_dir, "stats.json"))

        record = {
            "agent": facts.get("agent", "unknown"),
            "model": facts.get("model", "unknown"),
            "round": facts.get("round"),
            "passed": bool(verdict.get("passed")),
            "stages": verdict.get("stages", []),
            "__path": path,
        }
        for field in STAT_FIELDS:
            record[field] = stats.get(field)
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


def summarize(runs):
    # Key by (agent, model) so the same model run by different agents (pi,
    # swe-mini, ...) is reported separately rather than silently merged.
    by_key = {}
    for run in runs:
        key = (run.get("agent", "unknown"), run["model"])
        entry = by_key.setdefault(
            key, {"runs": 0, "passed": 0, "stage_failures": {}, "_runs": []}
        )
        entry["runs"] += 1
        entry["_runs"].append(run)
        if run["passed"]:
            entry["passed"] += 1
        else:
            for name in failed_stage_names(run):
                entry["stage_failures"][name] = entry["stage_failures"].get(name, 0) + 1

    summary = {"agents": {}}
    for (agent, model), stats in by_key.items():
        rs = stats["_runs"]
        pass_at_1 = stats["passed"] / stats["runs"] if stats["runs"] else None
        compacted = sum(1 for r in rs if r.get("compacted"))
        summary["agents"].setdefault(agent, {})[model] = {
            "runs": stats["runs"],
            "passed": stats["passed"],
            "pass@1": pass_at_1,
            "stage_failures": stats["stage_failures"],
            # Mean run cost/effort across the rounds (None when no stats.json).
            "avg_total_tokens": _mean([r.get("total_tokens") for r in rs]),
            "avg_turns": _mean([r.get("turns") for r in rs]),
            "avg_steps": _mean([r.get("steps") for r in rs]),
            "avg_peak_context": _mean([r.get("peak_context") for r in rs]),
            "compacted_runs": compacted,
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
