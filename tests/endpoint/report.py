#!/usr/bin/env python3
"""Merge L0 and L1 results into one table (stdout and tests/endpoint/last-run.md)."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="run dir holding l0.json and <agent>.json")
    ap.add_argument("--meta", required=True, help="free-text header (endpoint, model, steps, date)")
    a = ap.parse_args()
    out = []
    p = lambda s="": out.append(s)  # noqa: E731

    p("# Endpoint test result")
    p()
    p(a.meta)
    p()
    l0 = None
    f = os.path.join(a.dir, "l0.json")
    if os.path.isfile(f):
        l0 = json.load(open(f))
        p("## L0 protocol probes")
        p()
        p("| probe | status | detail |")
        p("|---|---|---|")
        for r in l0["probes"]:
            p("| %s | %s | %s |" % (r["probe"], r["status"], r["detail"].replace("|", "/")))
        p()

    agents = sorted(fn[:-5] for fn in os.listdir(a.dir) if fn.endswith(".json") and fn != "l0.json")
    tot_in = tot_out = 0
    if agents:
        names = ["trace", "stats", "model_calls", "tool_calls", "usage", "end_reason", "parser_errors", "thinking"]
        p("## L1 agents on one tiled-gemm round")
        p()
        p("| agent | " + " | ".join(names) + " | tokens in | tokens out |")
        p("|---|" + "---|" * (len(names) + 2))
        details = []
        for ag in agents:
            d = json.load(open(os.path.join(a.dir, ag + ".json")))
            if d.get("skipped"):
                p("| %s | %s |  |  |" % (ag, " | ".join(["SKIP"] * len(names))))
                details.append("- %s: skipped, %s" % (ag, d["skipped"]))
                continue
            by = {c["check"]: c for c in d["checks"]}
            t = d.get("totals") or {}
            tin, tout = t.get("in"), t.get("out")
            tot_in += tin or 0
            tot_out += tout or 0
            p("| %s | %s | %s | %s |" % (ag, " | ".join(by[n]["status"] if n in by else "-" for n in names),
                                       "n/a" if tin is None else "{:,}".format(tin), "n/a" if tout is None else "{:,}".format(tout)))
            for c in d["checks"]:
                if c["status"] in ("WARN", "FAIL"):
                    details.append("- %s / %s / %s: %s" % (ag, c["check"], c["status"], c["detail"]))
        p()
        if details:
            p("### Details for WARN, FAIL and SKIP")
            p()
            out.extend(details)
            p()
    l0_in = l0["tokens_in"] if l0 else 0
    l0_out = l0["tokens_out"] if l0 else 0
    p("## Token cost of this run")
    p()
    p("| part | in | out |")
    p("|---|---|---|")
    p("| L0 probes | {:,} | {:,} |".format(l0_in, l0_out))
    p("| L1 agents | {:,} | {:,} |".format(tot_in, tot_out))
    p("| **total** | **{:,}** | **{:,}** |".format(l0_in + tot_in, l0_out + tot_out))
    p()
    p("Input tokens are summed over every model call (the prompt is re-sent each call), as in `.eval/stats.json`.")
    text = "\n".join(out) + "\n"
    print(text)
    open(os.path.join(HERE, "last-run.md"), "w").write(text)
    failed = (l0 and any(r["status"] == "FAIL" for r in l0["probes"])) or any(
        c["status"] == "FAIL" for ag in agents for c in json.load(open(os.path.join(a.dir, ag + ".json"))).get("checks", []))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
