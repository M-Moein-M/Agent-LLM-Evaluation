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


def load_runs(traces_root):
    """One record per round: verdict + identity from the co-located facts.json."""
    pattern = os.path.join(traces_root, "**", ".eval", "pass_result.json")
    runs = []
    for path in glob(pattern, recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                verdict = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        facts = {}
        facts_path = os.path.join(os.path.dirname(path), "facts.json")
        try:
            with open(facts_path, "r", encoding="utf-8") as handle:
                facts = json.load(handle)
        except (OSError, json.JSONDecodeError):
            pass

        runs.append(
            {
                "model": facts.get("model", "unknown"),
                "round": facts.get("round"),
                "passed": bool(verdict.get("passed")),
                "stages": verdict.get("stages", []),
                "__path": path,
            }
        )
    return runs


def failed_stage_names(run):
    """Stages that actually failed (excludes skipped)."""
    return [
        s.get("name")
        for s in run.get("stages", [])
        if not s.get("passed") and not s.get("skipped")
    ]


def summarize(runs):
    by_model = {}
    for run in runs:
        model = run["model"]
        entry = by_model.setdefault(model, {"runs": 0, "passed": 0, "stage_failures": {}})
        entry["runs"] += 1
        if run["passed"]:
            entry["passed"] += 1
        else:
            for name in failed_stage_names(run):
                entry["stage_failures"][name] = entry["stage_failures"].get(name, 0) + 1

    summary = {"models": {}}
    for model, stats in by_model.items():
        pass_at_1 = stats["passed"] / stats["runs"] if stats["runs"] else None
        summary["models"][model] = {
            "runs": stats["runs"],
            "passed": stats["passed"],
            "pass@1": pass_at_1,
            "stage_failures": stats["stage_failures"],
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
