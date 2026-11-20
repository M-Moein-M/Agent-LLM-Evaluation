#!/usr/bin/env python3
"""Extract the rounds of the Hugging Face traces dataset (Parquet) into a folder trace-vis can browse.

    pip install pyarrow
    python3 hf_extract.py --in hf-traces/data --out hf-eval-traces [--problem tiled-gemm] [--limit 50]
    ./run.sh -w ../../hf-eval-traces --runs results/runs.jsonl

Writes, under --out:
    results/runs.jsonl                                    one row per round (what trace-vis indexes)
    traces/<problem>/<harness>/<model>/<campaign>/round-N/
        agent.hf.json     the round's messages and metadata (read by parsers._parse_hf)
        solution.json     the agent's solution.json, when the row has one
        changed_diff.txt  the diff of the agent's code changes, when the row has one

Only this script needs pyarrow. trace-vis itself stays stdlib only.
"""
import argparse
import json
import os
import re
import sys

META_KEYS = ("id", "problem", "harness", "model", "campaign", "round", "passed", "fail_class",
             "end_reason", "end_severity", "turns", "peak_context", "total_input_tokens",
             "total_output_tokens", "n_messages", "n_tool_calls", "n_truncated_outputs",
             "has_reasoning", "reasoning_format", "clean", "harness_version",
             "harness_version_source", "prompt_source")
READ_COLUMNS = META_KEYS + ("stages_json", "system_prompt", "solution_json", "final_diff", "messages")


def _safe(part):
    """One path component: no separators, never empty."""
    return re.sub(r"[^\w.+-]", "_", str(part)) or "_"


def _parquet_files(path):
    if os.path.isfile(path):
        return [path]
    files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet"))
    if not files:
        sys.exit("error: no .parquet files under %s" % path)
    return files


def _wanted(row, a):
    return ((not a.problem or row["problem"] in a.problem)
            and (not a.harness or row["harness"] in a.harness)
            and (not a.model or row["model"] in a.model)
            and (a.passed is None or bool(row["passed"]) == a.passed))


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _runs_row(row, round_rel):
    try:
        stages = json.loads(row.get("stages_json") or "[]")
    except ValueError:
        stages = []
    in_tok, out_tok = row.get("total_input_tokens"), row.get("total_output_tokens")
    return {
        "problem": row["problem"], "gpu": "NVIDIA A40",
        "campaign": "/".join(("traces", _safe(row["problem"]), _safe(row["harness"]),
                              _safe(row["model"]), _safe(row["campaign"]))),
        "agent": row["harness"], "model": row["model"], "model_raw": row["model"],
        "round": row["round"], "passed": bool(row["passed"]), "stages": stages,
        "__path": "%s/.eval/pass_result.json" % round_rel,
        "total_input": in_tok, "total_output": out_tok,
        "total_tokens": (in_tok or 0) + (out_tok or 0), "turns": row.get("turns"),
        "steps": None, "peak_context": row.get("peak_context"), "compacted": None,
        "end_reason": row.get("end_reason"), "end_severity": row.get("end_severity"),
        "fail_class": row.get("fail_class"), "fail_detail": None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="a .parquet file or a directory of them")
    ap.add_argument("--out", required=True, help="output folder (created if missing)")
    ap.add_argument("--problem", action="append", help="keep only this problem (repeatable)")
    ap.add_argument("--harness", action="append", help="keep only this harness (repeatable)")
    ap.add_argument("--model", action="append", help="keep only this model (repeatable)")
    ap.add_argument("--passed", choices=("yes", "no"), help="keep only passed or only failed rounds")
    ap.add_argument("--limit", type=int, help="stop after this many rounds")
    a = ap.parse_args(argv)
    a.passed = None if a.passed is None else a.passed == "yes"

    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("error: pyarrow is required for the extraction (pip install pyarrow). "
                 "The viewer itself needs nothing.")

    runs, n = [], 0
    for fpath in _parquet_files(a.src):
        pf = pq.ParquetFile(fpath)
        cols = [c for c in READ_COLUMNS if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=16, columns=cols):
            for row in batch.to_pylist():
                if not _wanted(row, a):
                    continue
                if a.limit and n >= a.limit:
                    break
                round_rel = "/".join(("traces", _safe(row["problem"]), _safe(row["harness"]),
                                      _safe(row["model"]), _safe(row["campaign"]),
                                      "round-%d" % row["round"]))
                rd = os.path.join(a.out, round_rel)
                doc = {"format": "hf-trace-v1",
                       "meta": {k: row.get(k) for k in META_KEYS},
                       "system_prompt": row.get("system_prompt"),
                       "messages": row.get("messages") or []}
                _write(os.path.join(rd, "agent.hf.json"), json.dumps(doc, ensure_ascii=False))
                if row.get("solution_json"):
                    _write(os.path.join(rd, "solution.json"), row["solution_json"])
                if row.get("final_diff"):
                    _write(os.path.join(rd, "changed_diff.txt"), row["final_diff"])
                runs.append(_runs_row(row, round_rel))
                n += 1
            else:
                continue
            break
        if a.limit and n >= a.limit:
            break

    _write(os.path.join(a.out, "results", "runs.jsonl"),
           "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in runs))
    print("extracted %d rounds to %s" % (n, a.out), file=sys.stderr)
    print("browse: tools/trace-vis/run.sh -w %s --runs results/runs.jsonl" % a.out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
