#!/usr/bin/env python3
"""Print pass@1 and token use per (problem, agent, model) from traces/.

    python3 tools/summarize.py [--traces traces] [--all-campaigns] [--problem P] [--agent A]

Reads traces/<problem>/<agent>/<model>/<campaign>/round-N/.eval/{pass_result,stats}.json
(written by checks/check_pass.py and checks/trace_stats.py). By default only the
latest campaign of each (problem, agent, model) is counted. Field reference:
docs/results-schema.md. For aggregation with failure classes and figures see
tools/collect_results.py and reference-results/analysis/.
"""
import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default="traces")
    ap.add_argument("--all-campaigns", action="store_true", help="pool every campaign instead of the latest")
    ap.add_argument("--problem")
    ap.add_argument("--agent")
    a = ap.parse_args()

    cells = defaultdict(lambda: defaultdict(list))  # (problem, agent, model) -> campaign -> [round dict]
    pattern = os.path.join(a.traces, "*", "*", "*", "*", "round-*")
    for rd in sorted(glob.glob(pattern)):
        problem, agent, model, campaign = rd.split(os.sep)[-5:-1]
        if (a.problem and problem != a.problem) or (a.agent and agent != a.agent):
            continue
        res = load(os.path.join(rd, ".eval", "pass_result.json"))
        if res is None:
            continue  # round never reached the checker (see docs/gotchas.md)
        st = load(os.path.join(rd, ".eval", "stats.json")) or {}
        cells[(problem, agent, model)][campaign].append({
            "passed": bool(res.get("passed")),
            "in": st.get("total_input"), "out": st.get("total_output"),
            "turns": st.get("turns"), "end": st.get("end_reason"),
        })
    if not cells:
        sys.exit("no rounds with .eval/pass_result.json under %s" % a.traces)

    rows = []
    for key, camps in sorted(cells.items()):
        picked = camps if a.all_campaigns else {max(camps): camps[max(camps)]}
        rounds = [r for rs in picked.values() for r in rs]
        passed = sum(r["passed"] for r in rounds)
        def med(field):
            v = [r[field] for r in rounds if isinstance(r[field], (int, float))]
            return statistics.median(v) if v else None
        ends = defaultdict(int)
        for r in rounds:
            ends[r["end"] or "?"] += 1
        rows.append((key, len(picked), passed, len(rounds), med("in"), med("out"), med("turns"), dict(ends)))

    def fmt(x):
        return "-" if x is None else ("%.0fk" % (x / 1000) if x >= 10000 else "%.0f" % x)

    print("%-22s %-10s %-34s %5s %9s %9s %9s %6s  %s" % (
        "problem", "agent", "model", "camps", "pass@1", "in(med)", "out(med)", "turns", "end reasons"))
    for (problem, agent, model), nc, p, n, tin, tout, turns, ends in rows:
        print("%-22s %-10s %-34s %5d %4d/%-4d %9s %9s %6s  %s" % (
            problem, agent, model[:34], nc, p, n, fmt(tin), fmt(tout), fmt(turns),
            ",".join("%s:%d" % kv for kv in sorted(ends.items()))))


if __name__ == "__main__":
    main()
