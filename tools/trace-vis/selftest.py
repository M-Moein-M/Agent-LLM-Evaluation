#!/usr/bin/env python3
"""Build-time gate for trace-vis. Runs offline, stdlib only.

Three groups of checks:

  schemas   synthesize a minimal round dir for each of the 4 trajectory schemas
            and assert the parsers normalize them identically (tool results
            stitched onto their calls, no orphans, no warnings). Fixtures are
            synthetic on purpose: %post has no traces/ bound.
  curation  the "important runs" scope (curation.py) applies each of its rules —
            round floor, A40 only, latest campaign wins, identity merges,
            excluded campaigns and rounds.
  assets    every script/stylesheet index.html references actually ships, the
            vendored libs are real, and the JS/CSS/Python kind lists agree.
  lint      a bracket/quote balance pass over the JS (no node in the image, and
            a syntax error there blanks the whole page)

  --workspace PATH  additionally parse real traces, a few per agent. Meant for
                    running on the host, not during the build.
  --js              additionally run jstest.js (payload rendering) under node,
                    or under `apptainer exec docker://node:22-slim` if there is
                    no node. Never part of the build — the image has no node.
"""
import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
sys.path.insert(0, HERE)

import parsers  # noqa: E402

FAILURES = []
CHECKS = [0]


def check(cond, label, detail=""):
    CHECKS[0] += 1
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}" + (f" — {detail}" if detail else ""))
        FAILURES.append(label)
    return cond


# --------------------------------------------------------------- fixtures ---

def _write(root, name, payload, jsonl=False):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    return d


# verbatim from traces/RabbitCT-LolaOMP/pi/qwen35bselfhosted_.../20260816-103214/round-4
PI_OVERFLOW_ERROR = (
    '500: {"message":"litellm.MidStreamFallbackError: litellm.APIConnectionError: '
    'APIConnectionError: OpenAIException - Input length (220211 tokens) exceeds the '
    'maximum allowed length (215684 tokens). Use a shorter input or enable '
    '--allow-auto-truncate.. Received Model Group=Qwen/Qwen3.6-35B-A3B-FP8\\n'
    'Available Model Group Fallbacks=None","type":null,"param":null,"code":"500"}'
)


def fixture_pi(root):
    d = os.path.join(root, "pi", "round-1")
    os.makedirs(d)
    lines = [
        {"type": "session", "id": "s1", "cwd": "/work", "timestamp": 1000},
        {"type": "message_end", "message": {
            "role": "user", "timestamp": 1000,
            "content": [{"type": "text", "text": "# Task\nMake it fast."}]}},
        {"type": "message_end", "message": {
            "role": "assistant", "model": "test-model", "timestamp": 1001, "content": [
                {"type": "thinking", "thinking": "Consider tiling."},
                {"type": "text", "text": "I'll inspect the kernel."},
                {"type": "toolCall", "id": "c1", "name": "bash",
                 "arguments": {"command": "ls -la"}}]}},
        {"type": "message_end", "message": {
            "role": "toolResult", "toolCallId": "c1", "isError": False, "timestamp": 1002,
            "content": [{"type": "text", "text": "kernel.cu\nMakefile"}]}},
        # how a run that outgrew the context window actually ends: the reason
        # lives only here, and the retry verdict arrives after it
        {"type": "agent_end", "willRetry": False, "messages": [{
            "role": "assistant", "content": [], "stopReason": "error", "timestamp": 1003,
            "errorMessage": PI_OVERFLOW_ERROR}]},
        {"type": "auto_retry_end", "success": False, "attempt": 3,
         "finalError": PI_OVERFLOW_ERROR},
    ]
    with open(os.path.join(d, "agent.jsonl"), "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    return d, "pi"


def fixture_swemini(root):
    d = os.path.join(root, "swe-mini", "round-1")
    os.makedirs(d)
    doc = {
        "trajectory_format": "mini-swe-agent-1",
        "info": {"config": {"model": {"model_name": "test-model"}},
                 "model_stats": {"api_calls": 2}},
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "# Task\nMake it fast."},
            {"role": "assistant", "content": "Inspecting.", "tool_calls": [
                {"id": "c1", "function": {"name": "bash",
                                          "arguments": json.dumps({"command": "ls -la"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "kernel.cu\nMakefile",
             "extra": {"returncode": 0, "timestamp": 1002}},
            {"role": "exit", "extra": {"exit_status": "submitted"}},
        ],
    }
    with open(os.path.join(d, "agent.traj.json"), "w") as f:
        json.dump(doc, f)
    return d, "swe-mini"


def fixture_claude(root):
    d = os.path.join(root, "claude", "round-1")
    os.makedirs(d)
    lines = [
        {"type": "system", "subtype": "init", "session_id": "s1", "cwd": "/work",
         "model": "test-model", "tools": ["Bash"], "claude_code_version": "0.0.0"},
        {"type": "user", "message": {"role": "user", "content": "# Task\nMake it fast."}},
        {"type": "assistant", "message": {"role": "assistant", "model": "test-model", "content": [
            {"type": "thinking", "thinking": "Consider tiling."},
            {"type": "text", "text": "Inspecting."},
            {"type": "tool_use", "id": "c1", "name": "bash", "input": {"command": "ls -la"}}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "is_error": False,
             "content": [{"type": "text", "text": "kernel.cu\nMakefile"}]}]}},
        {"type": "result", "subtype": "success", "is_error": False,
         "duration_ms": 1234, "num_turns": 2, "result": "done"},
    ]
    with open(os.path.join(d, "agent.stream.jsonl"), "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    return d, "claude"


def fixture_kimi(root):
    d = os.path.join(root, "kimi", "round-1")
    os.makedirs(d)
    # first line must carry no "type" key — that's how detect_kind tells kimi
    # apart from claude, both of which use agent.stream.jsonl
    lines = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "# Task\nMake it fast."},
        {"role": "assistant", "content": "Inspecting.", "tool_calls": [
            {"id": "c1", "function": {"name": "bash",
                                      "arguments": json.dumps({"command": "ls -la"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "kernel.cu\nMakefile"},
        {"role": "meta", "type": "usage", "total_tokens": 42},
    ]
    with open(os.path.join(d, "agent.stream.jsonl"), "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    return d, "kimi"


# verbatim from traces/tiled-gemm/kimi/Qwen_Qwen3.6-35B-A3B-FP8/20260815-230556/round-4
KIMI_MAX_STEPS_ERROR = (
    'error: failed to run prompt: loop.max_steps_exceeded: Turn exceeded '
    'maxSteps=120. If max_steps_per_turn is too small, raise it in config.toml '
    '(loop_control.max_steps_per_turn), or run "/update-config" to update it, '
    'then "/reload".\nSee log: /problem/.kimi-code-home/logs/kimi-code.log\n'
)


def fixture_kimi_cutoff(root):
    """kimi's own trace never records why a run stopped — a harness-side cutoff
    (step cap, provider error, dropped connection) only ever lands in the
    sibling agent.stderr.log, not agent.stream.jsonl."""
    d = os.path.join(root, "kimi", "round-2")
    os.makedirs(d)
    lines = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "# Task\nMake it fast."},
        {"role": "assistant", "content": "Trying WMMA.", "tool_calls": [
            {"id": "c1", "function": {"name": "bash",
                                      "arguments": json.dumps({"command": "make"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "error: identifier undefined"},
    ]
    with open(os.path.join(d, "agent.stream.jsonl"), "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    with open(os.path.join(d, "agent.stderr.log"), "w") as f:
        f.write(KIMI_MAX_STEPS_ERROR)
    return d, "kimi"


FIXTURES = [fixture_pi, fixture_swemini, fixture_claude, fixture_kimi]


def test_schemas():
    print("schemas")
    with tempfile.TemporaryDirectory() as root:
        for make in FIXTURES:
            d, want = make(root)
            kind, fname = parsers.detect_kind(d)
            if not check(kind == want, f"{want}: detect_kind", f"got {kind!r} from {fname!r}"):
                continue
            res = parsers.parse_trace(d)
            evs = res["events"]
            check(not res["warnings"], f"{want}: parses without warnings", str(res["warnings"]))
            check(len(evs) >= 3, f"{want}: produced events", f"got {len(evs)}")

            calls = [e for e in evs if e["kind"] == "tool_call"]
            check(len(calls) == 1, f"{want}: one tool_call", f"got {len(calls)}")
            if calls:
                c = calls[0]
                check(c["tool_name"] == "bash", f"{want}: tool name", repr(c["tool_name"]))
                check(c["tool_output"] and "kernel.cu" in c["tool_output"],
                      f"{want}: tool result stitched onto its call", repr(c["tool_output"]))
                check(isinstance(c["tool_input"], dict) and c["tool_input"].get("command") == "ls -la",
                      f"{want}: tool input decoded", repr(c["tool_input"]))
                check(c["is_error"] is False, f"{want}: not flagged as error")

            orphans = [e for e in evs if e["kind"] == "log" and "orphan" in (e["text"] or "")]
            check(not orphans, f"{want}: no orphan tool results", str(orphans))
            check(any(e["kind"] == "assistant_text" for e in evs), f"{want}: has assistant text")
            check(all(e.get("idx") == i for i, e in enumerate(evs)), f"{want}: events indexed")
            check(all(e["kind"] in KINDS_PY for e in evs), f"{want}: only known kinds",
                  str(sorted({e["kind"] for e in evs} - KINDS_PY)))

            # pagination contract the UI relies on
            page = parsers.slice_events(res, 0, 2)
            check(page["returned"] == 2 and page["has_more"] and page["total_filtered"] == len(evs),
                  f"{want}: slice_events paginates", json.dumps(
                      {k: page[k] for k in ("returned", "has_more", "total_filtered")}))
            only = parsers.slice_events(res, 0, 50, {"tool_call"})
            check(only["total_filtered"] == len(calls), f"{want}: slice_events filters by kind")


def test_run_end():
    """pi records why a run stopped only in its trailing agent_end envelope.

    A context overflow is the case that matters: the timeline otherwise just
    ends mid-conversation and reads like the model gave up.
    """
    print("run end (pi)")
    with tempfile.TemporaryDirectory() as root:
        d, _ = fixture_pi(root)
        evs = parsers.parse_trace(d)["events"]
        ev = evs[-1] if evs else {}
        check(ev.get("kind") == "run_end", "the pi trace ends with a run_end event",
              repr(ev.get("kind")))
        check(ev.get("end_reason") == "context_overflow", "the overflow is classified",
              repr(ev.get("end_reason")))
        check((ev.get("end_tokens_used"), ev.get("end_tokens_limit")) == (220211, 215684),
              "both token counts are pulled out of the provider message",
              f"{ev.get('end_tokens_used')} / {ev.get('end_tokens_limit')}")
        check("215,684" in (ev.get("text") or "") and "over" in (ev.get("text") or ""),
              "the summary says how far over the limit it went", repr(ev.get("text")))
        check("retried 3" in (ev.get("text") or ""),
              "the failed retries are folded in (auto_retry_end lands after agent_end)",
              repr(ev.get("text")))
        check(ev.get("is_error") and ev.get("end_severity") == "error",
              "an overflow is flagged as an error")
        check("Input length" in (ev.get("tool_output") or ""),
              "the raw provider message is kept for the disclosure")

    cases = [
        ("stop", None, "completed", "ok"),
        ("length", None, "output_limit", "warn"),
        (None, None, "cut_off", "warn"),
        ("error", '429: {"message":"Rate limit exceeded for model_per_key: x"}',
         "rate_limit", "error"),
        ("error", "Request timed out.", "timeout", "error"),
        ("error", '400: {"message":"litellm.BadRequestError: OpenAIException - Requested '
                  "token count exceeds the model's maximum context length of 256000 tokens. "
                  'You requested a total of 257647 tokens"}', "context_overflow", "error"),
        ("error", '400: {"message":"max_completion_tokens is too large: 256326.'
                  'This model supports at most 256000 completion tokens."}',
         "output_limit", "warn"),
        ("error", "400 status code (no body)", "api_error", "error"),
    ]
    for stop, err, reason, sev in cases:
        got = parsers._pi_end_reason(stop, err)[0]
        check(got == reason, f"stopReason {stop!r} -> {reason}", f"got {got}")
        ev = parsers._pi_run_end({"stop": stop, "error": err, "will_retry": False,
                                  "ts": None}, None)
        check(ev["end_severity"] == sev, f"{reason} is a {sev}", ev["end_severity"])
    reason, used, limit, what = parsers._pi_end_reason(
        "error", '400: {"message":"Requested token count exceeds the model\'s maximum '
        'context length of 256000 tokens. You requested a total of 257647 tokens"}')
    check((used, limit) == (257647, 256000),
          "used/limit stay in that order when the message states the total first")
    check("total" in (what or ""),
          "a provider that reports input+completion is not called 'input tokens'", repr(what))
    check(parsers._pi_run_end(None, None) is None,
          "a trace with no agent_end gets no run_end event")
    check(parsers._provider_message('500: {"message":"boom","code":"500"}') == "boom",
          "the human part is unwrapped from the nested json error")
    check(parsers._provider_message("Request timed out.") == "Request timed out.",
          "a bare error message is left alone")

    print("run end (kimi)")
    with tempfile.TemporaryDirectory() as root:
        d, _ = fixture_kimi_cutoff(root)
        evs = parsers.parse_trace(d)["events"]
        ev = evs[-1] if evs else {}
        check(ev.get("kind") == "run_end", "the kimi trace ends with a run_end event",
              repr(ev.get("kind")))
        check(ev.get("end_reason") == "max_steps_exceeded",
              "a harness-side step cap is read from the sibling stderr log, "
              "not the (silent) jsonl trace", repr(ev.get("end_reason")))
        check("120-step cap" in (ev.get("text") or ""),
              "the step count is pulled out of the stderr message", repr(ev.get("text")))
        check(ev.get("end_severity") == "warn", "a step cap is a warn, not an error")

        # round-1: no stderr error, and the trace ends on plain assistant text
        # with no pending tool call — a clean completion, no harness cutoff
        d2 = os.path.join(root, "kimi", "round-1")
        os.makedirs(d2)
        with open(os.path.join(d2, "agent.stream.jsonl"), "w") as f:
            for ln in [{"role": "user", "content": "# Task\nMake it fast."},
                       {"role": "assistant", "content": "Done, it's tiled now."}]:
                f.write(json.dumps(ln) + "\n")
        evs2 = parsers.parse_trace(d2)["events"]
        ev2 = evs2[-1] if evs2 else {}
        check(ev2.get("end_reason") == "completed",
              "no stderr error + last message is assistant text reads as a clean completion",
              repr(ev2.get("end_reason")))
        check(ev2.get("end_severity") == "ok", "a clean completion is not flagged")

    with tempfile.TemporaryDirectory() as root:
        # kimi failing before it writes any output at all (e.g. rate limited on
        # the very first request): agent.stream.jsonl is empty, so the claude/
        # kimi disambiguation has nothing to peek at. Falls back to stderr's
        # "error: failed to run prompt:" line, which only the kimi CLI writes.
        d = os.path.join(root, "round-1")
        os.makedirs(d)
        open(os.path.join(d, "agent.stream.jsonl"), "w").close()
        with open(os.path.join(d, "agent.stderr.log"), "w") as f:
            f.write("error: failed to run prompt: provider.rate_limit: 429 rate limited\n"
                    "See log: /problem/.kimi-code-home/logs/kimi-code.log\n")
        check(parsers.detect_kind(d) == ("kimi", "agent.stream.jsonl"),
              "an empty jsonl still detects as kimi via the stderr fallback")
        ev = parsers.parse_trace(d)["events"][-1]
        check(ev.get("end_reason") == "rate_limit",
              "the rate limit is classified from stderr alone", repr(ev.get("end_reason")))

        # both files empty: genuinely no signal left to recover from
        d2 = os.path.join(root, "round-2")
        os.makedirs(d2)
        open(os.path.join(d2, "agent.stream.jsonl"), "w").close()
        open(os.path.join(d2, "agent.stderr.log"), "w").close()
        check(parsers.detect_kind(d2) == ("claude", "agent.stream.jsonl"),
              "an empty jsonl with no stderr signal falls back to claude — "
              "there is nothing left to tell the two apart")

    print("run end (claude)")
    # The distinction that matters: a run the harness cut at EVAL_MAX_TURNS and
    # a run the model ended by itself look identical in the timeline — both stop
    # right after a tool result. Only the final `result` record tells them apart.
    with tempfile.TemporaryDirectory() as root:
        d, _ = fixture_claude(root)                 # ends success/is_error:false
        clean = parsers.parse_trace(d)["events"][-1]
        check(clean.get("kind") == "run_end" and clean.get("end_reason") == "completed",
              "a claude trace that ended on its own reads as completed",
              repr(clean.get("end_reason")))
        check(clean.get("end_severity") == "ok" and not clean.get("is_error"),
              "a clean completion is not flagged")
        check(not clean.get("tool_output"),
              "a completed run does not repeat its final answer as a provider message")

        d2 = os.path.join(root, "claude-maxturns", "round-1")
        os.makedirs(d2)
        with open(os.path.join(d2, "agent.stream.jsonl"), "w") as f:
            for ln in [{"type": "system", "subtype": "init", "session_id": "s1"},
                       {"type": "assistant", "message": {"role": "assistant", "content": [
                           {"type": "tool_use", "id": "c1", "name": "bash",
                            "input": {"command": "make"}}]}},
                       {"type": "user", "message": {"role": "user", "content": [
                           {"type": "tool_result", "tool_use_id": "c1", "is_error": True,
                            "content": [{"type": "text", "text": "ptxas error"}]}]}},
                       {"type": "result", "subtype": "error_max_turns", "is_error": True,
                        "terminal_reason": "max_turns", "num_turns": 121,
                        "stop_reason": "tool_use", "result": None,
                        "errors": ["Reached maximum number of turns (120)"]}]:
                f.write(json.dumps(ln) + "\n")
        ev = parsers.parse_trace(d2)["events"][-1]
        check(ev.get("end_reason") == "max_turns",
              "error_max_turns is read as the harness cutting the run off",
              repr(ev.get("end_reason")))
        check("121 turns" in (ev.get("text") or ""),
              "the turn count is stated", repr(ev.get("text")))
        check("never acted on" in (ev.get("text") or ""),
              "stopping on stop_reason tool_use says the last result went unused",
              repr(ev.get("text")))
        check(ev.get("end_severity") == "warn",
              "a turn cap is a warn — the harness stopped it, the provider did not")
        check("Reached maximum number of turns" in (ev.get("tool_output") or ""),
              "the harness's own error line is kept for the disclosure")

        # killed outright (walltime, SIGKILL): the stream just stops, no result
        d3 = os.path.join(root, "claude-killed", "round-1")
        os.makedirs(d3)
        with open(os.path.join(d3, "agent.stream.jsonl"), "w") as f:
            for ln in [{"type": "system", "subtype": "init", "session_id": "s1"},
                       {"type": "assistant", "message": {"role": "assistant", "content": [
                           {"type": "tool_use", "id": "c1", "name": "bash",
                            "input": {"command": "pkill -f runner"}}]}}]:
                f.write(json.dumps(ln) + "\n")
        ev = parsers.parse_trace(d3)["events"][-1]
        check(ev.get("end_reason") == "cut_off",
              "no result record at all means the process was killed mid-run",
              repr(ev.get("end_reason")))
        check("killed" in (ev.get("text") or ""),
              "the summary says so in words", repr(ev.get("text")))

    # every provider failure claude reported in this corpus, by its result text
    cases = [
        ({"subtype": "success", "is_error": True, "terminal_reason": "api_error",
          "result": "API Error: Request rejected (429) · litellm.RateLimitError: "
                    "Rate limit exceeded for model_per_key: x"}, "rate_limit", "error"),
        ({"subtype": "success", "is_error": True, "terminal_reason": "blocking_limit",
          "result": "Prompt is too long"}, "context_overflow", "error"),
        ({"subtype": "success", "is_error": True, "terminal_reason": "api_error",
          "result": "API Error: Content block is not a thinking block"},
         "thinking_block", "error"),
        ({"subtype": "success", "is_error": True, "terminal_reason": "api_error",
          "result": "API Error: Connection closed mid-response."},
         "connection_error", "error"),
        ({"subtype": "success", "is_error": True, "terminal_reason": "api_error",
          "result": "Request timed out"}, "timeout", "error"),
        ({"subtype": "success", "is_error": True, "terminal_reason": "api_error",
          "result": "You've hit your session limit · resets 7am (UTC)"},
         "rate_limit", "error"),
        ({"subtype": "success", "is_error": True, "terminal_reason": "api_error",
          "result": "API Error: 500 something nobody has seen before"},
         "api_error", "error"),
        ({"subtype": "success", "is_error": False, "terminal_reason": "completed",
          "stop_reason": "max_tokens"}, "output_limit", "warn"),
    ]
    for res, reason, sev in cases:
        ev = parsers._claude_run_end(res)
        check(ev["end_reason"] == reason,
              f"claude result {str(res.get('result'))[:34]!r} -> {reason}",
              ev["end_reason"])
        check(ev["end_severity"] == sev, f"claude {reason} is a {sev}", ev["end_severity"])

    vllm = {"subtype": "success", "is_error": True, "terminal_reason": "api_error",
            "api_error_status": 400,
            "result": "API Error: 400 litellm.ContextWindowExceededError: "
                      "Hosted_vllmException - This model's maximum context length is "
                      "262144 tokens. However, you requested 128000 output tokens and "
                      "your prompt contains at least 134145 input tokens, for a total "
                      "of at least 262145 tokens."}
    ev = parsers._claude_run_end(vllm)
    check((ev["end_tokens_used"], ev["end_tokens_limit"]) == (262145, 262144),
          "vLLM's own context wording yields both counts",
          f"{ev['end_tokens_used']} / {ev['end_tokens_limit']}")
    check("in total" in ev["text"],
          "a total that includes the requested completion is not called input tokens",
          repr(ev["text"]))
    check(ev.get("tool_output", "").count("400") == 1,
          "the http status is not repeated when the message already states it",
          repr(ev.get("tool_output"))[:120])
    ev = parsers._claude_run_end({"subtype": "success", "is_error": True,
                                  "terminal_reason": "api_error",
                                  "api_error_status": 529,
                                  "result": "API Error: overloaded"})
    check("HTTP 529" in (ev.get("tool_output") or ""),
          "a status the message omits is carried into the disclosure",
          repr(ev.get("tool_output")))


    print("run end (swe-mini)")
    # swe-mini names its ending in `role:"exit"` -> extra.exit_status. It used to
    # render as a log line reading "session exit: ContextWindowExceededError",
    # which said nothing about how far over the window the run went.
    with tempfile.TemporaryDirectory() as root:
        d, _ = fixture_swemini(root)                # exits "submitted"
        ev = parsers.parse_trace(d)["events"][-1]
        check(ev.get("kind") == "run_end" and ev.get("end_reason") == "completed",
              "a swe-mini trajectory ends with a run_end, not a log line",
              f"{ev.get('kind')} / {ev.get('end_reason')}")
        check(ev.get("end_severity") == "ok" and "Submitted" in (ev.get("text") or ""),
              "a submitted run says so and is not flagged", repr(ev.get("text")))

        # a trajectory killed before it could write its exit message
        d2 = os.path.join(root, "swemini-killed", "round-1")
        os.makedirs(d2)
        with open(os.path.join(d2, "agent.traj.json"), "w") as f:
            json.dump({"info": {}, "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "working"}]}, f)
        ev = parsers.parse_trace(d2)["events"][-1]
        check(ev.get("end_reason") == "cut_off" and "killed" in (ev.get("text") or ""),
              "no exit message means the process was killed mid-run",
              repr(ev.get("end_reason")))

    ctx = ("litellm.ContextWindowExceededError: litellm.BadRequestError: "
           "ContextWindowExceededError: OpenAIException - The input (258274 tokens) "
           "is longer than the model's context length (256000 tokens).\n"
           "model=Qwen/Qwen3.8-27B. context_window_fallbacks=None.")
    ev = parsers._swemini_run_end(
        {"content": ctx, "extra": {"exit_status": "ContextWindowExceededError",
                                   "exception_str": ctx}}, {})
    check(ev["end_reason"] == "context_overflow" and ev["end_severity"] == "error",
          "litellm's ContextWindowExceededError is classified as an overflow",
          ev["end_reason"])
    check((ev["end_tokens_used"], ev["end_tokens_limit"]) == (258274, 256000),
          "both token counts come out of litellm's wording",
          f"{ev['end_tokens_used']} / {ev['end_tokens_limit']}")
    check("2,274 over" in ev["text"], "the summary says how far over it went",
          repr(ev["text"]))
    check("litellm" in (ev["tool_output"] or ""),
          "the raw exception is kept for the disclosure")

    info = {"config": {"agent": {"step_limit": 120, "cost_limit": 5.0,
                                 "max_consecutive_format_errors": 3}},
            "model_stats": {"api_calls": 120, "instance_cost": 0.0}}
    ev = parsers._swemini_run_end({"content": "LimitsExceeded",
                                   "extra": {"exit_status": "LimitsExceeded"}}, info)
    check(ev["end_reason"] == "limits_exceeded" and ev["end_severity"] == "warn",
          "a step/cost cap is the harness stopping the run, so it is a warn",
          f"{ev['end_reason']} / {ev['end_severity']}")
    check("all 120 steps" in ev["text"],
          "the cap that was actually hit is named", repr(ev["text"]))
    check(not ev["tool_output"],
          "a status with no message behind it has nothing to disclose")

    cost = {"config": {"agent": {"step_limit": 500, "cost_limit": 5.0}},
            "model_stats": {"api_calls": 40, "instance_cost": 5.0}}
    ev = parsers._swemini_run_end({"extra": {"exit_status": "LimitsExceeded"}}, cost)
    check("$5.00 of a $5.00 budget" in ev["text"],
          "a run that spent its budget says that instead of the step count",
          repr(ev["text"]))

    exit_fmt = {"role": "exit", "content": "RepeatedFormatError",
                "extra": {"exit_status": "RepeatedFormatError"}}
    ev = parsers._swemini_run_end(exit_fmt, info)
    check(ev["end_reason"] == "format_error" and "3 in a row" in ev["text"],
          "repeated malformed tool calls name the configured cap", repr(ev["text"]))

    # RepeatedFormatError is the symptom. A model that is cut off mid-tool-call
    # (finish_reason "length") emits no tool call either, and calling that a
    # formatting problem hides a token budget that ran out.
    def fmt_interrupt(finish, prompt, completion):
        return {"role": "user", "content": "Tool call error:",
                "extra": {"interrupt_type": "FormatError",
                          "response": {"choices": [{"finish_reason": finish}],
                                       "usage": {"prompt_tokens": prompt,
                                                 "completion_tokens": completion}}}}

    full = [fmt_interrupt("length", 255430, 568)] * 3 + [exit_fmt]
    ev = parsers._swemini_run_end(exit_fmt, info, full)
    check(ev["end_reason"] == "context_overflow",
          "no room left to answer in is an overflow, not a format error",
          ev["end_reason"])
    check("255,430-token prompt" in ev["text"] and "3x in a row" in ev["text"],
          "the note says what filled the window and how often", repr(ev["text"]))
    check("exit status RepeatedFormatError" in ev["text"],
          "the harness's own word for the ending is still shown")

    capped = [fmt_interrupt("length", 4000, 32000)] * 3 + [exit_fmt]
    check(parsers._swemini_run_end(exit_fmt, info, capped)["end_reason"] == "output_limit",
          "a long reply after a short prompt is an output cap, not the window")

    genuine = [fmt_interrupt("stop", 9000, 400)] * 3 + [exit_fmt]
    check(parsers._swemini_run_end(exit_fmt, info, genuine)["end_reason"] == "format_error",
          "a model that answered and just never called a tool IS a format error")
    mixed = [fmt_interrupt("length", 9000, 400), fmt_interrupt("stop", 9000, 400), exit_fmt]
    check(parsers._swemini_run_end(exit_fmt, info, mixed)["end_reason"] == "format_error",
          "one truncated answer among well-formed ones does not reclassify the run")
    check(parsers._swemini_run_end(exit_fmt, info, [exit_fmt])["end_reason"] == "format_error",
          "a trajectory with no interrupt records left keeps the harness's verdict")

    bad = ("litellm.BadRequestError: OpenAIException - This model only supports "
           "single tool-calls at once!")
    ev = parsers._swemini_run_end({"content": bad,
                                   "extra": {"exit_status": "BadRequestError",
                                             "exception_str": bad}}, {})
    check(ev["end_reason"] == "api_error" and ev["end_severity"] == "error",
          "an exit class we have no rule for still reads as a provider error",
          ev["end_reason"])
    check("exit status BadRequestError" in ev["text"],
          "the harness's own word for the ending stays visible", repr(ev["text"]))


def _jsonl(path, lines):
    with open(path, "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")


def _kimi_round(root, name, lines):
    d = os.path.join(root, "kimi", name)
    os.makedirs(d)
    _jsonl(os.path.join(d, "agent.stream.jsonl"),
           [{"role": "user", "content": "# Task\nMake it fast."}] + lines)
    return d


def _kimi_step_end(finish, context, output):
    return {"type": "context.append_loop_event", "event": {
        "type": "step.end", "finishReason": finish,
        "usage": {"inputOther": context, "inputCacheRead": 0,
                  "inputCacheCreation": 0, "output": output}}}


def test_limits():
    """Output-cap, hang and compaction reporting shared by trace_stats and the viewer."""
    # trace_stats lives in checks/, which the image does not ship: the viewer
    # checks run everywhere, the stats checks only from the source tree.
    try:
        import trace_stats
    except ImportError:
        trace_stats = None
    print("limits (kimi output cap, hang; capped calls, nudges, compaction)")
    resume = {"role": "meta", "type": "session.resume_hint", "session_id": "s1"}
    with tempfile.TemporaryDirectory() as root:
        # kimi ends the round on a capped call, yet the stream closes like a clean
        # completion; only the wire log's step.end finishReason shows the cap
        d = _kimi_round(root, "capped", [
            {"role": "assistant", "content": "Writing it all out..."}, resume])
        os.makedirs(os.path.join(d, "agent.kimi-wire"))
        _jsonl(os.path.join(d, "agent.kimi-wire", "main.wire.jsonl"), [
            _kimi_step_end("tool_calls", 20000, 500),
            {"type": "compaction.started"},
            _kimi_step_end("max_tokens", 22604, 10000)])
        ev = parsers.parse_trace(d)["events"][-1]
        check(ev.get("end_reason") == "output_limit",
              "kimi: a capped last call in the wire log is output_limit, not completed",
              repr(ev.get("end_reason")))

        # a background task kept kimi alive after its final answer; the end
        # watch in solve.sh ended it and left agent.end_watch.log
        d2 = _kimi_round(root, "watched", [
            {"role": "assistant", "content": "Done, it's tiled now."}])
        with open(os.path.join(d2, "agent.end_watch.log"), "w") as f:
            f.write("2026-09-26T04:47:07 hung_after_final: agent alive 600s after "
                    "its first reply with no tool call, ending it\n")
        ev2 = parsers.parse_trace(d2)["events"][-1]
        check(ev2.get("end_reason") == "hung_after_final",
              "kimi: an end-watch hang is hung_after_final", repr(ev2.get("end_reason")))
        check(ev2.get("end_severity") == "error", "a hang is flagged as an error")

        # same hang before the watch existed: 4h solve timeout, final reply with
        # no tool call, and a command kimi moved to the background
        d3 = _kimi_round(root, "timeout", [
            {"role": "assistant", "content": "Running it.", "tool_calls": [
                {"id": "c1", "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": "next_step: The task now runs in the background."},
            {"role": "assistant", "content": "Done."}])
        open(os.path.join(d3, ".solve_timeout"), "w").close()
        ev3 = parsers.parse_trace(d3)["events"][-1]
        check(ev3.get("end_reason") == "hung_after_final",
              "kimi: solve timeout + backgrounded command + final reply is hung_after_final",
              repr(ev3.get("end_reason")))
        os.remove(os.path.join(d3, ".solve_timeout"))
        ev3b = parsers.parse_trace(d3)["events"][-1]
        check(ev3b.get("end_reason") == "completed",
              "kimi: the same round without a timeout is a clean completion",
              repr(ev3b.get("end_reason")))

        if trace_stats is None:
            print("  skip trace_stats checks (checks/ not on the path)")
            return
        st = trace_stats.parse_kimi(trace_stats.Path(d) / "agent.stream.jsonl")
        check((st["end_reason"], st["capped_calls"], st["capped_ctx"])
              == ("output_limit", 1, [32604]),
              "trace_stats kimi: output_limit, 1 capped call at input + output",
              repr((st["end_reason"], st["capped_calls"], st["capped_ctx"])))
        check(st["compaction_points"] == [1],
              "trace_stats kimi: compaction comes from the wire record, after call 1",
              repr(st["compaction_points"]))
        st2 = trace_stats.parse_kimi(trace_stats.Path(d2) / "agent.stream.jsonl")
        check((st2["end_reason"], st2["capped_calls"]) == ("hung_after_final", None),
              "trace_stats kimi: hang without a wire log, capped calls unknown",
              repr((st2["end_reason"], st2["capped_calls"])))

        # pi: capped calls from stopReason length, context includes both cache
        # tiers, compaction from compaction_start (a cache hit is not a drop)
        dp = os.path.join(root, "pi", "round-1")
        os.makedirs(dp)

        def pi_call(inp, cache, out, stop):
            return {"type": "message_end", "message": {
                "role": "assistant", "content": [], "stopReason": stop,
                "usage": {"input": inp, "cacheRead": cache, "cacheWrite": 0,
                          "output": out, "totalTokens": inp + cache + out}}}
        _jsonl(os.path.join(dp, "agent.jsonl"), [
            pi_call(10000, 0, 100, "toolUse"),
            pi_call(500, 10000, 64000, "length"),
            {"type": "compaction_start", "reason": "threshold"},
            pi_call(3000, 0, 100, "toolUse")])
        sp = trace_stats.parse_pi(trace_stats.Path(dp) / "agent.jsonl")
        check((sp["capped_calls"], sp["capped_ctx"]) == (1, [74500]),
              "trace_stats pi: 1 capped call, context counts cached input",
              repr((sp["capped_calls"], sp["capped_ctx"])))
        check((sp["peak_context"], sp["total_input"]) == (10500, 23500),
              "trace_stats pi: peak and input include cacheRead",
              repr((sp["peak_context"], sp["total_input"])))
        check(sp["compaction_points"] == [2],
              "trace_stats pi: one compaction after call 2, none for the cache hit",
              repr(sp["compaction_points"]))

        # claude: re-prompts after a thinking-only reply are counted, capped
        # calls stay unknown (stream-json has no per-call stop reason)
        dc = os.path.join(root, "claude", "round-1")
        os.makedirs(dc)

        def cl_call(mid):
            return {"type": "assistant", "message": {"id": mid, "content": [
                {"type": "thinking", "thinking": "..."}],
                "usage": {"input_tokens": 100, "output_tokens": 10}}}
        _jsonl(os.path.join(dc, "agent.stream.jsonl"), [
            {"type": "system", "subtype": "init", "session_id": "s1"},
            cl_call("m1"),
            {"type": "user", "message": {"role": "user", "content":
                "[Your previous response had no visible output. Please continue.]"}},
            {"type": "system", "subtype": "compact_boundary"},
            cl_call("m2"),
            {"type": "result", "subtype": "success", "is_error": False, "num_turns": 1}])
        sc = trace_stats.parse_claude(trace_stats.Path(dc) / "agent.stream.jsonl")
        check((sc["turns"], sc["empty_reply_nudges"], sc["capped_calls"])
              == (2, 1, None),
              "trace_stats claude: 2 calls, 1 empty-reply nudge, capped calls unknown",
              repr((sc["turns"], sc["empty_reply_nudges"], sc["capped_calls"])))
        check(sc["compaction_points"] == [1],
              "trace_stats claude: compaction from compact_boundary",
              repr(sc["compaction_points"]))


def test_fail_class():
    """One fine-grained reason per failing round, and never the word "build".

    failure_class.py is shared with checks/, so this gate covers the reports as
    well as the viewer. The precedence is the whole point: a graded measurement
    beats the harness's artifact finding, and "the agent never wrote an
    artifact" loses to how the agent's session actually ended.
    """
    print("fail class")
    fc = parsers.fc

    def artifact(facts, build_log=None, run_log=None):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "facts.json"), "w") as fh:
                json.dump(facts, fh)
            for name, text in (("build.log", build_log), ("run.log", run_log)):
                if text is not None:
                    with open(os.path.join(d, name), "w") as fh:
                        fh.write(text)
            return fc.classify_artifact(d)

    ok = {"build": {"status": 0}, "run": {"status": 0}}
    bad_build = {"build": {"status": 2}, "run": {"status": 1}}

    cases = [
        ({"failure_reason": "No solution.json produced"}, None, None, "no_solution"),
        ({"failure_code": "no_solution"}, None, None, "no_solution"),
        ({"failure_reason": "Incomplete solution.json (missing build/run)"},
         None, None, "incomplete_solution"),
        ({"failure_reason": "something measure.sh has not said yet"},
         None, None, "measure_refused"),
        (bad_build, "make: Stopping after creating config.mk\n", None, "make_config_stop"),
        (bad_build, "make: nvcc: Command not found\n", None, "toolchain_missing"),
        (bad_build, "make: *** No rule to make target 'x.o'.  Stop.\n", None, "make_no_rule"),
        (bad_build, "gemm.o: undefined reference to `foo'\n", None, "link_error"),
        (bad_build, "ptxas error   : Entry function uses too much shared data\n",
         None, "ptxas_error"),
        (bad_build, "gemm.cu(31): error: identifier \"blockDim\" is undefined\n",
         None, "compile_error"),
        (bad_build, "cc: cannot open util.h: No such file or directory\n",
         None, "missing_file"),
        (bad_build, "linking\nfinished\n", None, "build_error"),
        ({"build": {"status": 124}}, None, None, "build_timeout"),
        ({"build": {"status": 0}, "run": {"status": 124}}, None, None, "run_timeout"),
        ({"build": {"status": 0}, "run": {"status": 139}}, None,
         "warming up\nSegmentation fault\n", "run_error"),
        ({"build": {"status": 0}, "run": {"status": 1}}, None,
         "Unknown algorithm: LolaOMPGPU\nAvailable algorithms:\n  LolaBunny\n",
         "run_error"),
        (ok, None, None, None),
    ]
    for facts, blog, rlog, want in cases:
        got, detail = artifact(facts, blog, rlog)
        check(got == want, f"artifact -> {want}", f"got {got!r} ({detail!r})")
    check(artifact(bad_build, "gemm.cu(31): error: bad\n")[1].startswith("gemm.cu(31)"),
          "the detail quotes the offending line, not the whole log")
    # a runner prints its usage banner AFTER the thing that went wrong, so the
    # last line of run.log is routinely the wrong quote
    got = artifact({"build": {"status": 0}, "run": {"status": 1}}, None,
                   "./run-bench.sh: illegal option -- c\nUsage: ./run-bench.sh -s <size>\n"
                   "Environment: TOOLCHAIN (default: CLANG)\n")[1]
    check(got == "./run-bench.sh: illegal option -- c",
          "run.log is quoted at its first error line, not its last line", repr(got))

    def stages(**kw):
        return [{"name": n, "passed": p, "detail": ""} for n, p in kw.items()]

    passing = stages(build=True, run=True, mse=True)
    check(fc.fail_class(True, passing, "completed")[0] is None,
          "a passing round has no fail class")

    with tempfile.TemporaryDirectory() as d:
        json.dump(ok, open(os.path.join(d, "facts.json"), "w"))
        klass, _ = fc.fail_class(False, stages(build=True, run=True, mse=False),
                                 "completed", d)
        check(klass == "mse", "a failed graded stage IS the reason", klass)
        klass, _ = fc.fail_class(False, stages(build=True, run=True, mse=False),
                                 "context_overflow", d)
        check(klass == "mse", "a measurement beats how the session ended", klass)

    with tempfile.TemporaryDirectory() as d:
        json.dump({"failure_reason": "No solution.json produced"},
                  open(os.path.join(d, "facts.json"), "w"))
        empty = stages(build=False, run=False)
        for end, want in (("context_overflow", "context_overflow"),
                          ("format_error", "format_error"),
                          ("max_turns", "max_turns"),
                          ("completed", "no_solution"),
                          (None, "no_solution")):
            klass, _ = fc.fail_class(False, empty, end, d)
            check(klass == want, f"no artifact + ended {end} -> {want}", klass)

    with tempfile.TemporaryDirectory() as d:
        json.dump(bad_build, open(os.path.join(d, "facts.json"), "w"))
        open(os.path.join(d, "build.log"), "w").write("gemm.cu(9): error: nope\n")
        klass, _ = fc.fail_class(False, stages(build=False), "context_overflow", d)
        check(klass == "compile_error",
              "a real build log beats the run-end reason (the artifact exists)", klass)

    with tempfile.TemporaryDirectory() as d:
        json.dump(ok, open(os.path.join(d, "facts.json"), "w"))
        klass, _ = fc.fail_class(False, stages(build=False), "completed", d)
        check(klass == "build_error",
              "a stage that fails with clean receipts is still not called 'build'", klass)

    for name in fc.FAIL_CLASS_LABEL:
        check(name not in ("build", "run"), f"{name!r} is not a bare stage name")
    for slugs in (fc.FACTS_FAILURE_CODE.values(),
                  [s for _, s in fc.BUILD_PATTERNS],
                  fc.GRADED_STAGES):
        for slug in slugs:
            check(slug in fc.FAIL_CLASS_LABEL, f"{slug} has a human label")


def test_caps():
    print("caps")
    big = "x" * (parsers.MAX_TEXT + 500)
    capped, trunc = parsers._cap(big)
    check(len(capped) == parsers.MAX_TEXT and trunc, "long text is capped and flagged")
    small, trunc = parsers._cap("short")
    check(small == "short" and not trunc, "short text passes through")
    obj, _ = parsers._cap({"a": 1})
    check(isinstance(obj, str) and '"a"' in obj, "non-string values are JSON-encoded")
    check(parsers._norm_ts(1.7e12) == 1.7e9, "millisecond timestamps normalize to seconds")
    check(parsers._norm_ts("nope") is None, "unparseable timestamps drop to None")

    with tempfile.TemporaryDirectory() as d:
        res = parsers.parse_trace(d)
        check(res["events"] == [] and res["warnings"], "empty round dir degrades gracefully")


# ----------------------------------------------------------------- assets ---

KINDS_PY = {"system", "user", "assistant_text", "thinking", "tool_call", "log",
            "run_end"}


def test_assets():
    print("assets")
    index = os.path.join(STATIC, "index.html")
    if not check(os.path.isfile(index), "static/index.html exists"):
        return
    html = open(index, encoding="utf-8").read()
    refs = re.findall(r'(?:src|href)="(/[^"]+)"', html)
    check(len(refs) >= 6, "index.html references its assets", f"found {refs}")
    for ref in refs:
        p = os.path.join(STATIC, ref.lstrip("/"))
        check(os.path.isfile(p) and os.path.getsize(p) > 0, f"asset present: {ref}")

    for lib, marker, floor in (("vendor/marked.min.js", "marked", 20000),
                               ("vendor/purify.min.js", "DOMPurify", 15000)):
        p = os.path.join(STATIC, lib)
        ok = os.path.isfile(p) and os.path.getsize(p) > floor
        check(ok, f"vendored lib looks real: {lib}",
              "missing or too small — the UI must work with no network")
        if ok:
            check(marker in open(p, encoding="utf-8", errors="replace").read(2000000),
                  f"vendored lib is the right library: {lib}")

    # the event-kind vocabulary is duplicated across three files; drift there is
    # invisible until an event silently vanishes from the timeline
    tl = open(os.path.join(STATIC, "js", "timeline.js"), encoding="utf-8").read()
    m = re.search(r"FILTER_KINDS\s*=\s*\[([^\]]*)\]", tl)
    if check(bool(m), "timeline.js declares FILTER_KINDS"):
        kinds_js = set(re.findall(r"'([a-z_]+)'", m.group(1)))
        check(kinds_js == KINDS_PY, "JS filter kinds match the parser's kinds",
              f"js={sorted(kinds_js)} py={sorted(KINDS_PY)}")
        labels = set(re.findall(r"^\s*([a-z_]+):\s*'", tl, re.M))
        check(KINDS_PY <= labels, "every kind has a display label",
              f"missing {sorted(KINDS_PY - labels)}")

    # tool payloads must go through the decoder, not straight into a <pre>:
    # a raw dump shows the escape "\n" where the shell output should break
    check("payloadBlocks" in tl and "window.TV.payloadBlocks" in tl,
          "timeline.js decodes tool payloads (payloadBlocks, exported for jstest.js)")

    # the agent view groups campaign -> problem; every class it emits needs a rule,
    # or the waves collapse into one undifferentiated list of tables
    app = open(os.path.join(STATIC, "js", "app.js"), encoding="utf-8").read()
    check("groupWaves" in app and "wavesBlock" in app,
          "app.js groups the agent view by campaign wave")

    css = open(os.path.join(STATIC, "css", "style.css"), encoding="utf-8").read()
    check(".pkey" in css, "payload field labels are styled")
    for cls in ("wave", "wave-head", "wave-prob"):
        check(f".{cls}" in css and f'"{cls}' in app,
              f"agent-view wave class is emitted and styled: .{cls}")
    for k in KINDS_PY:
        check(f".ev.{k}::before" in css, f"timeline dot styled for kind: {k}")
    check(not re.search(r":\s*#[0-9a-fA-F]*[g-zG-Z]", css), "no malformed hex colors in CSS")


# ------------------------------------------------------------------- lint ---

# tokens after which a '/' starts a regex literal rather than a division
_RE_OK_CHARS = set("(,=:[!&|?{};+-*%~^<>")
_RE_OK_WORDS = {"return", "typeof", "case", "in", "of", "new", "delete", "void", "instanceof"}


def js_balance(src):
    """Bracket/quote balance scan. A lint, not a parser — but it catches the
    realistic failure (a stray brace or quote from hand-editing) with no node."""
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    i, n = 0, len(src)
    line = 1
    prev_sig = ""          # last significant char, for regex-vs-divide
    prev_word = ""
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c in " \t\r":
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            i = src.find("\n", i)
            if i < 0:
                break
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            if j < 0:
                return f"unterminated block comment opened at line {line}"
            line += src.count("\n", i, j)
            i = j + 2
            continue
        if c in "\"'`":
            quote, j = c, i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "\n" and quote != "`":
                    return f"unterminated string at line {line}"
                if src[j] == quote:
                    break
                j += 1
            if j >= n:
                return f"unterminated string at line {line}"
            line += src.count("\n", i, j)
            i = j + 1
            prev_sig, prev_word = quote, ""
            continue
        if c == "/" and (prev_sig in _RE_OK_CHARS or prev_sig == "" or prev_word in _RE_OK_WORDS):
            j = i + 1
            in_class = False
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "\n":
                    return f"unterminated regex at line {line}"
                if src[j] == "[":
                    in_class = True
                elif src[j] == "]":
                    in_class = False
                elif src[j] == "/" and not in_class:
                    break
                j += 1
            if j >= n:
                return f"unterminated regex at line {line}"
            i = j + 1
            while i < n and src[i].isalpha():   # flags
                i += 1
            prev_sig, prev_word = "/", ""
            continue
        if c in "([{":
            stack.append((c, line))
        elif c in ")]}":
            if not stack:
                return f"unbalanced {c!r} at line {line}"
            open_c, open_line = stack.pop()
            if open_c != pairs[c]:
                return f"{c!r} at line {line} closes {open_c!r} from line {open_line}"
        if c.isalnum() or c == "_":
            k = i
            while k < n and (src[k].isalnum() or src[k] == "_"):
                k += 1
            prev_word = src[i:k]
            prev_sig = src[k - 1]
            i = k
            continue
        prev_sig, prev_word = c, ""
        i += 1
    if stack:
        return f"unclosed {stack[-1][0]!r} from line {stack[-1][1]}"
    return None


def test_lint():
    print("lint")
    jsdir = os.path.join(STATIC, "js")
    for name in sorted(os.listdir(jsdir)):
        if not name.endswith(".js"):
            continue
        src = open(os.path.join(jsdir, name), encoding="utf-8").read()
        err = js_balance(src)
        check(err is None, f"js/{name} balances", err or "")
        check("window.TV" in src, f"js/{name} uses the TV namespace")

    # every TV.<name> the app calls must be exported by some module
    app = open(os.path.join(jsdir, "app.js"), encoding="utf-8").read()
    exported = set()
    for name in ("markdown.js", "timeline.js"):
        exported |= set(re.findall(r"window\.TV\.(\w+)\s*=", open(os.path.join(jsdir, name), encoding="utf-8").read()))
    used = set(re.findall(r"window\.TV\.(\w+)", app)) - {"TV"}
    check(used <= exported, "app.js only uses exported TV helpers",
          f"undefined: {sorted(used - exported)}")

    # self-check the linter, so a broken linter can't pass everything
    check(js_balance("function f() { return 1; }") is None, "linter accepts valid js")
    check(js_balance("function f() { return 1;") is not None, "linter rejects an unclosed brace")
    check(js_balance("var s = 'oops;") is not None, "linter rejects an unterminated string")
    check(js_balance("var r = /['\"]/g; var x = (1 + 2);") is None,
          "linter handles quotes inside regex literals")
    check(js_balance("var a = 4 / 2; var b = 6 / 3;") is None, "linter handles division")


# --------------------------------------------------------------- curation ---

def _row(problem, agent, model, ts, rnd, gpu="NVIDIA A40", model_dir=None):
    """A minimal index row (see indexer._read_runs_jsonl) for curation tests."""
    md = model_dir or model
    return {"run_id": f"{problem}/{agent}/{model}/{ts}/round-{rnd}",
            "problem": problem, "agent": agent, "model": model,
            "campaign": ts, "campaign_rel": f"traces/{problem}/{agent}/{md}/{ts}",
            "round": rnd, "gpu": gpu, "has_eval": True}


def _camp(problem, agent, model, ts, n, **kw):
    return [_row(problem, agent, model, ts, i, **kw) for i in range(1, n + 1)]


def test_curation():
    print("curation")
    import curation

    keep = _camp("P", "pi", "M", "20260701-000000", 5)
    check(curation.curated_run_ids(keep) == {r["run_id"] for r in keep},
          "a plain 5-round A40 campaign is kept whole")

    check(curation.curated_run_ids(_camp("P", "pi", "M", "20260701-000000", 2)) == set(),
          "campaigns under 3 rounds are dropped")
    check(len(curation.curated_run_ids(_camp("P", "pi", "M", "20260701-000000", 3))) == 3,
          "3-round campaigns are kept (Kimi-K3's floor)")

    cpu = _camp("P", "pi", "M", "20260728-000000", 5, gpu="nvidia-smi not found")
    check(curation.curated_run_ids(cpu) == set(), "non-A40 rounds are out of scope")

    old = _camp("P", "pi", "M", "20260701-000000", 5)
    new = _camp("P", "pi", "M", "20260809-000000", 5)
    got = curation.curated_run_ids(old + new)
    check(got == {r["run_id"] for r in new}, "the latest campaign supersedes earlier ones",
          f"kept {sorted(got)[:2]}")

    # the same weights under two ids must not each keep their own campaign
    bare = _camp("P", "pi", "gemma-4-31B-it", "20260801-000000", 5, model_dir="gemmaselfhosted")
    fp8 = _camp("P", "pi", "gemma-4-31B-it-FP8-block", "20260701-000000", 5)
    got = curation.curated_run_ids(bare + fp8)
    check(got == {r["run_id"] for r in bare}, "model aliases collapse to one pairing")

    # claude-pro is the claude harness, so it competes with claude for latest
    pro = _camp("P", "claude-pro", "claude-opus-4-8", "20260801-000000", 5)
    cla = _camp("P", "claude", "claude-opus-4-8", "20260701-000000", 5)
    got = curation.curated_run_ids(pro + cla)
    check(got == {r["run_id"] for r in pro}, "claude-pro and claude share one harness key")

    excl = next(iter(curation.EXCLUDE_CAMPAIGNS)).split("/")
    rows = _camp(excl[0], excl[1], excl[2], excl[3], 5)
    check(curation.curated_run_ids(rows) == set(), "EXCLUDE_CAMPAIGNS entries are dropped")

    (prob, agent, model), bad = next(iter(curation.ARTIFACT_ROUNDS.items()))
    rows = _camp(prob, agent, model, "20260805-003151", 10)
    got = curation.curated_run_ids(rows)
    check(got == {r["run_id"] for r in rows if r["round"] not in bad},
          "artifact rounds are dropped from an otherwise kept campaign",
          f"expected {len(rows) - len(bad)} kept, got {len(got)}")

    check(curation.normalize_scope(None) == curation.DEFAULT_SCOPE
          and curation.normalize_scope("all") == "all"
          and curation.normalize_scope("curated") == "curated"
          and curation.normalize_scope("nonsense") == curation.DEFAULT_SCOPE,
          "scope parameter falls back to the default scope")


# -------------------------------------------------------------- real data ---

def test_workspace(ws, per_agent=2):
    print(f"workspace {ws}")
    from indexer import TraceIndex
    idx = TraceIndex(ws).build()
    check(len(idx.runs) > 0, "index built runs", f"{len(idx.runs)} runs")
    check(not idx.warnings, "index built without warnings", str(idx.warnings[:3]))

    import curation
    check(0 < len(idx.curated) <= len(idx.runs),
          f"curated scope covers {len(idx.curated)} of {len(idx.runs)} runs")
    check(idx.curated <= set(idx.runs), "every curated run_id exists in the index")
    ws_excl = curation.workspace_exclude_campaigns(ws)
    check(ws_excl is None or ws_excl == curation.EXCLUDE_CAMPAIGNS,
          "EXCLUDE_CAMPAIGNS matches checks/report_tables.py",
          f"workspace={sorted(ws_excl or [])} here={sorted(curation.EXCLUDE_CAMPAIGNS)}")

    an_agent = sorted(idx.by_agent)[0]
    v_cur = idx.agent_view(an_agent, "curated")
    v_all = idx.agent_view(an_agent, "all")
    check(v_cur["n_runs"] <= v_all["n_runs"] and v_all["n_runs"] == len(
              [r for p in idx.by_agent[an_agent].values() for r in p]),
          f"agent view honours scope ({an_agent}: {v_cur['n_runs']} curated / {v_all['n_runs']} all)")
    s_cur, s_all = idx.summary("curated"), idx.summary("all")
    check(s_cur["totals"]["n_runs"] == len(idx.curated)
          and s_all["totals"]["n_runs"] == len(idx.runs),
          "summary totals match each scope",
          f"{s_cur['totals']['n_runs']}/{s_all['totals']['n_runs']}")
    check(sum(a["n_runs"] for a in s_cur["agents"]) == len(idx.curated),
          "curated per-agent counts add up to the curated total")
    check(sum(m["n_runs"] for m in s_cur["models"]) == len(idx.curated),
          "curated per-model counts add up to the curated total")

    a_model = sorted(idx.by_model)[0]
    mv = idx.model_view(a_model, "all")
    rounds = sum(len(c["runs"]) for a in mv["agents"]
                 for p in a["problems"] for c in p["campaigns"])
    check(mv["n_runs"] == rounds == len(
              [r for a in idx.by_model[a_model].values() for p in a.values() for r in p]),
          f"model view reaches every round ({a_model}: {mv['n_runs']} runs,"
          f" {len(mv['agents'])} agents)")
    check(all(r["model"] == a_model for a in mv["agents"] for p in a["problems"]
              for c in p["campaigns"] for r in c["runs"]),
          "model view only contains that model's runs")
    check(idx.model_view("no-such-model", "all") is None,
          "an unknown model has no view")

    for agent in sorted(idx.by_agent):
        seen = 0
        for problem in sorted(idx.by_agent[agent]):
            for rid in idx.by_agent[agent][problem]:
                if seen >= per_agent:
                    break
                path = idx.round_path(rid)
                if path is None:
                    continue
                kind, _ = parsers.detect_kind(path)
                if kind is None:
                    continue
                res = parsers.parse_trace(path)
                orphans = sum(1 for e in res["events"]
                              if e["kind"] == "log" and "orphan" in (e["text"] or ""))
                check(res["events"] and not res["warnings"] and not orphans,
                      f"{agent}/{kind}: {res['stats']['total_events']} events, no orphans",
                      f"{rid} warnings={res['warnings']} orphans={orphans}")
                seen += 1
            if seen >= per_agent:
                break


# --------------------------------------------------------------- js (opt) ---

# Only reached with --js. The image has no node, so this is never a build gate;
# the first apptainer fallback run needs network to pull the image.
NODE_IMAGE = "docker://node:22-slim"


def _js_runner(readable):
    """An argv prefix that runs `node <file>`, with `readable` paths visible."""
    env = os.environ.get("TRACE_VIS_NODE")
    if env:
        return shlex.split(env)
    for exe in ("node", "nodejs"):
        found = shutil.which(exe)
        if found:
            return [found]
    apptainer = shutil.which("apptainer") or shutil.which("singularity")
    if apptainer:
        binds = []
        for p in readable:
            binds += ["-B", p]
        return [apptainer, "exec"] + binds + [NODE_IMAGE, "node"]
    return None


def _js_cases(workspace, out_path, per_bucket=3):
    """Real parsed events for jstest.js to render: JSON tool outputs, tool
    inputs holding multi-line text, and tagged log lines."""
    buckets = {
        "json_out": lambda e: (e.get("tool_output") or "").strip()[:1] in ("{", "["),
        "long_in": lambda e: isinstance(e.get("tool_input"), dict) and any(
            isinstance(v, str) and "\n" in v for v in e["tool_input"].values()),
        "tagged_log": lambda e: e["kind"] == "log" and (e.get("text") or "")[:1] == "[",
        "run_end": lambda e: e["kind"] == "run_end",
    }
    counts = dict.fromkeys(buckets, 0)
    cases = []
    pattern = os.path.join(workspace, "traces", "*", "*", "*", "*", "round-1")
    for round_dir in sorted(glob.glob(pattern)):
        if all(counts[b] >= per_bucket for b in buckets):
            break
        if parsers.detect_kind(round_dir)[0] is None:
            continue
        try:
            events = parsers.parse_trace(round_dir)["events"]
        except Exception:
            continue
        for name, match in buckets.items():
            if counts[name] >= per_bucket:
                continue
            for ev in events:
                if match(ev):
                    cases.append({"src": round_dir[len(workspace):], "ev": ev})
                    counts[name] += 1
                    break
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(cases, fh)
    return counts


def test_js(workspace=None):
    print("js")
    cases_path = os.path.join(tempfile.gettempdir(), f"tv-jscases-{os.getpid()}.json")
    readable = [HERE, tempfile.gettempdir()]
    if workspace:
        readable.append(os.path.abspath(workspace))
    runner = _js_runner(readable)
    if not check(runner is not None, "a javascript runtime is available",
                 "no node/nodejs/apptainer — set TRACE_VIS_NODE to a runner"):
        return

    cmd = list(runner) + [os.path.join(HERE, "jstest.js")]
    if workspace:
        counts = _js_cases(workspace, cases_path)
        if check(all(counts.values()), "collected real events for the js test", str(counts)):
            cmd.append(cases_path)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as exc:                                   # noqa: BLE001
        check(False, "jstest.js runs", f"{type(exc).__name__}: {exc}")
        return
    finally:
        if os.path.exists(cases_path):
            os.unlink(cases_path)
    out = (proc.stdout or "") + (proc.stderr or "")
    for line in out.splitlines():
        if line.strip():
            print("    " + line)
    check(proc.returncode == 0, "jstest.js passes",
          f"exit {proc.returncode} — see the output above")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", help="also parse real traces from this workspace")
    ap.add_argument("--js", action="store_true",
                    help="also run jstest.js, if a javascript runtime can be found")
    args = ap.parse_args(argv)

    test_schemas()
    test_run_end()
    test_limits()
    test_fail_class()
    test_caps()
    test_curation()
    test_assets()
    test_lint()
    if args.js:
        test_js(args.workspace)
    if args.workspace:
        test_workspace(args.workspace)

    print()
    if FAILURES:
        print(f"selftest: FAILED — {len(FAILURES)}/{CHECKS[0]} checks failed")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"selftest: all {CHECKS[0]} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
