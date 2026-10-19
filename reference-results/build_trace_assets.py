#!/usr/bin/env python3
"""Pack traces/ into one tar.zst per problem for upload as GitHub release assets.

    python3 reference-results/build_trace_assets.py --traces traces --out /path/to/assets \
        [--problems tiled-gemm,stream-compact] [--rules private-rules.tsv] [--level 10]

Only an allow-list of files per round is packed (the agent trace, its stderr, solution.json,
changed_diff.txt and the small `.eval/` evidence files). Workspace leftovers (binaries, volumes, core
dumps, profiler output, sources) are skipped: they are most of the 41 GB and carry no results.

--rules FILE: tab-separated `python-regex<TAB>replacement` lines applied to every packed file name and
to the content of every packed text file (use it to strip account names, host names, e-mail addresses
before publishing). Keep that file outside the repo. A line starting with `#` is a comment.

Writes <out>/traces-<problem>.tar.zst, <out>/traces-manifest.csv (one row per campaign) and
<out>/traces-assets.sha256. Needs the `zstd` binary.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile

AGENT_TRACES = ("agent.jsonl", "agent.traj.json", "agent.stream.jsonl", "agent.stderr.log",
                "solution.json", "changed_diff.txt")
AGENT_DIRS = ("agent.kimi-wire",)
EVAL_FILES = ("facts.json", "pass_result.json", "stats.json", "run.log", "build.log", "perf.log",
              "perf_paired.log", "gpu_usage.csv", "gpu_info.txt", "host_info.txt", "kern_agent.csv",
              "kern_cublas.csv")
MARKERS = (".solve_timeout", ".measure_timeout", ".no_trace")
MAX_FILE = 200 * 1024 * 1024  # a single kept file above this is skipped and reported


def load_rules(path):
    rules = []
    if not path:
        return rules
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        pat, _, rep = line.partition("\t")
        rules.append((re.compile(pat.encode()), rep.encode()))
    return rules


def scrub_bytes(data, rules):
    for rx, rep in rules:
        data = rx.sub(rep, data)
    return data


def round_files(rd):
    """Yield (relative name, absolute path) of allow-listed files in one round dir."""
    for n in AGENT_TRACES + MARKERS:
        p = os.path.join(rd, n)
        if os.path.isfile(p):
            yield n, p
    for d in AGENT_DIRS:
        dp = os.path.join(rd, d)
        if os.path.isdir(dp):
            for fn in sorted(os.listdir(dp)):
                if os.path.isfile(os.path.join(dp, fn)):
                    yield d + "/" + fn, os.path.join(dp, fn)
    for n in EVAL_FILES:
        p = os.path.join(rd, ".eval", n)
        if os.path.isfile(p):
            yield ".eval/" + n, p


def campaigns(traces, problem):
    base = os.path.join(traces, problem)
    for agent in sorted(os.listdir(base)):
        ap = os.path.join(base, agent)
        if not os.path.isdir(ap):
            continue
        for model in sorted(os.listdir(ap)):
            mp = os.path.join(ap, model)
            if not os.path.isdir(mp) or model == "endpoint-test":  # endpoint-test = tests/endpoint output
                continue
            for camp in sorted(os.listdir(mp)):
                cp = os.path.join(mp, camp)
                rounds = sorted(d for d in os.listdir(cp) if d.startswith("round-")) if os.path.isdir(cp) else []
                if rounds:
                    yield agent, model, camp, cp, rounds


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default="traces")
    ap.add_argument("--out", required=True)
    ap.add_argument("--problems")
    ap.add_argument("--rules")
    ap.add_argument("--level", type=int, default=10, help="zstd level")
    a = ap.parse_args()
    rules = load_rules(a.rules)
    os.makedirs(a.out, exist_ok=True)
    problems = a.problems.split(",") if a.problems else sorted(
        d for d in os.listdir(a.traces) if os.path.isdir(os.path.join(a.traces, d)) and not d.startswith(("_", ".")))
    manifest, skipped = [], []
    for problem in problems:
        if not os.path.isdir(os.path.join(a.traces, problem)):
            continue
        if not any(True for _ in campaigns(a.traces, problem)):
            continue
        asset = "traces-%s.tar.zst" % problem
        out_path = os.path.join(a.out, asset)
        zst = subprocess.Popen(["zstd", "-T0", "-%d" % a.level, "-q", "-f", "-o", out_path], stdin=subprocess.PIPE)
        tar = tarfile.open(fileobj=zst.stdin, mode="w|")
        for agent, model, camp, cp, rounds in campaigns(a.traces, problem):
            n_pass = n_eval = raw = 0
            for rnd in rounds:
                rd = os.path.join(cp, rnd)
                pr = os.path.join(rd, ".eval", "pass_result.json")
                if os.path.isfile(pr):
                    try:
                        n_pass += bool(json.load(open(pr)).get("passed"))
                        n_eval += 1
                    except ValueError:
                        pass
                for rel, abspath in round_files(rd):
                    size = os.path.getsize(abspath)
                    if size > MAX_FILE:
                        skipped.append((abspath, size))
                        continue
                    name = scrub_bytes(("%s/%s/%s/%s/%s/%s" % (problem, agent, model, camp, rnd, rel)).encode(), rules).decode()
                    data = scrub_bytes(open(abspath, "rb").read(), rules) if rules else open(abspath, "rb").read()
                    ti = tarfile.TarInfo("traces/" + name)
                    ti.size, ti.mtime, ti.mode = len(data), int(os.path.getmtime(abspath)), 0o644
                    tar.addfile(ti, io.BytesIO(data))
                    raw += len(data)
            m_model = scrub_bytes(model.encode(), rules).decode()
            manifest.append({"problem": problem, "agent": agent, "model": m_model, "campaign": camp,
                             "rounds": len(rounds), "graded": n_eval, "passed": n_pass,
                             "bytes_uncompressed": raw, "asset": asset})
        tar.close()
        zst.stdin.close()
        if zst.wait() != 0:
            sys.exit("zstd failed for %s" % asset)
        print("%s: %.1f MB" % (asset, os.path.getsize(out_path) / 1e6), flush=True)
    with open(os.path.join(a.out, "traces-manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]) if manifest else ["problem"])
        w.writeheader()
        w.writerows(manifest)
    with open(os.path.join(a.out, "traces-assets.sha256"), "w") as f:
        for asset in sorted({m["asset"] for m in manifest}):
            f.write("%s  %s\n" % (sha256_file(os.path.join(a.out, asset)), asset))
    for p, s in skipped:
        print("skipped (over %d MB): %s" % (MAX_FILE >> 20, p), file=sys.stderr)


if __name__ == "__main__":
    main()
