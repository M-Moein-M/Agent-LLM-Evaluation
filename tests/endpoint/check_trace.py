#!/usr/bin/env python3
"""L1 assertions on one agent round written by scripts/run_eval.sh.

    python3 tests/endpoint/check_trace.py --round DIR --agent pi [--reasoning] [--json OUT]

Reads the agent's raw trace and DIR/.eval/stats.json. The question is whether the
plumbing worked (parsing, tool calls, thinking, token accounting), not whether the
agent solved the problem, so a failed build/measure phase is ignored.
"""
import argparse
import glob
import json
import os
import sys

TRACE_FILES = {"pi": "agent.jsonl", "swe-mini": "agent.traj.json",
               "claude": "agent.stream.jsonl", "kimi": "agent.stream.jsonl"}
PARSER_ERRORS = ("RepeatedFormatError", "tool_use_failed", "Failed to parse", "JSONDecodeError",
                 "Unexpected token", "invalid_request_error", "tool call parser")
MARKUP = ("<tool_call>", "<function=", "<|tool_call", "[TOOL_CALLS]")


def jl(path):
    out = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def thinking_and_text(rd, agent):
    """(number of non-empty thinking blocks, assistant text with tool-call markup count)."""
    think = markup = 0
    if agent == "pi":
        for e in jl(os.path.join(rd, "agent.jsonl")):
            m = e.get("message") or {}
            if e.get("type") == "message_end" and m.get("role") == "assistant":
                for c in m.get("content") or []:
                    if c.get("type") == "thinking" and (c.get("thinking") or "").strip():
                        think += 1
                    if c.get("type") == "text" and any(t in (c.get("text") or "") for t in MARKUP):
                        markup += 1
    elif agent == "claude":
        for e in jl(os.path.join(rd, "agent.stream.jsonl")):
            if e.get("type") == "assistant":
                for c in (e.get("message") or {}).get("content") or []:
                    if c.get("type") == "thinking" and (c.get("thinking") or "").strip():
                        think += 1
                    if c.get("type") == "text" and any(t in (c.get("text") or "") for t in MARKUP):
                        markup += 1
    elif agent == "kimi":
        for w in glob.glob(os.path.join(rd, "agent.kimi-wire", "*.wire.jsonl")):
            for e in jl(w):
                part = ((e.get("event") or {}).get("part") or {})
                if part.get("type") == "think" and (part.get("think") or "").strip():
                    think += 1
        for e in jl(os.path.join(rd, "agent.stream.jsonl")):
            if e.get("role") == "assistant" and isinstance(e.get("content"), str) and any(t in e["content"] for t in MARKUP):
                markup += 1
    elif agent == "swe-mini":
        try:
            msgs = json.load(open(os.path.join(rd, "agent.traj.json"))).get("messages") or []
        except (OSError, ValueError):
            msgs = []
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            if (m.get("reasoning_content") or "").strip():
                think += 1
            if any(t in (m.get("content") or "") for t in MARKUP):
                markup += 1
    return think, markup


def check(rd, agent, reasoning):
    rows = []

    def add(cid, status, detail):
        rows.append({"check": cid, "status": status, "detail": detail})

    tf = os.path.join(rd, TRACE_FILES[agent])
    if not os.path.isfile(tf) or os.path.getsize(tf) == 0:
        add("trace", "FAIL", "%s missing or empty (the agent never produced a trace)" % TRACE_FILES[agent])
        return rows, {}
    add("trace", "PASS", "%s, %d bytes" % (TRACE_FILES[agent], os.path.getsize(tf)))

    sp = os.path.join(rd, ".eval", "stats.json")
    if not os.path.isfile(sp):
        add("stats", "FAIL", ".eval/stats.json missing (trace_stats.py did not run or could not parse the trace)")
        return rows, {}
    st = json.load(open(sp))
    add("stats", "PASS", "parsed as agent=%s source=%s" % (st.get("agent"), st.get("source")))

    turns, steps = st.get("turns") or 0, st.get("steps") or 0
    add("model_calls", "PASS" if turns >= 1 else "FAIL", "%d model call(s)" % turns)
    add("tool_calls", "PASS" if steps >= 1 else "FAIL",
        "%d tool execution(s) with results fed back" % steps if steps >= 1 else
        "no tool call executed (tool calling or its parsing is broken for this agent)")

    tin, tout = st.get("total_input"), st.get("total_output")
    if tin is None:
        add("usage", "WARN", "token usage not recorded for this agent/trace format")
    elif not (tin and tout):
        add("usage", "WARN", "usage is zero (the server returns no usage: token cost cannot be measured)")
    else:
        add("usage", "PASS", "in=%d out=%d over %d calls" % (tin, tout, turns))

    err = st.get("end_severity") == "error"
    add("end_reason", "FAIL" if err else "PASS",
        "%s%s" % (st.get("end_reason"), (": " + str(st.get("end_detail"))) if st.get("end_detail") else ""))

    log = os.path.join(rd, "agent.stderr.log")
    hits = []
    if os.path.isfile(log):
        txt = open(log, errors="replace").read()
        hits = [p for p in PARSER_ERRORS if p in txt]
    if st.get("end_stop_reason") == "RepeatedFormatError":
        hits.append("RepeatedFormatError (end_stop_reason)")
    add("parser_errors", "FAIL" if hits else "PASS", ("found: " + ", ".join(hits)) if hits else "none in agent.stderr.log")

    think, markup = thinking_and_text(rd, agent)
    if markup:
        add("tool_markup_in_text", "WARN", "%d assistant text block(s) contain tool-call markup (parser may be leaking calls)" % markup)
    if reasoning:
        add("thinking", "PASS" if think else "WARN",
            "%d thinking block(s)" % think if think else
            "model is declared reasoning but the %s trace holds no thinking (template disabled, or the proxy drops it)" % agent)
    else:
        add("thinking", "SKIP", "model not declared as reasoning")
    return rows, {"in": tin, "out": tout, "turns": turns, "steps": steps, "end_reason": st.get("end_reason")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", required=True)
    ap.add_argument("--agent", required=True, choices=sorted(TRACE_FILES))
    ap.add_argument("--reasoning", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows, totals = check(a.round, a.agent, a.reasoning)
    for r in rows:
        print("  %-5s %-20s %s" % (r["status"], r["check"], r["detail"]))
    if a.json:
        json.dump({"agent": a.agent, "checks": rows, "totals": totals}, open(a.json, "w"), indent=2)
    return 1 if any(r["status"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
