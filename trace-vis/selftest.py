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

KINDS_PY = {"system", "user", "assistant_text", "thinking", "tool_call", "log"}


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

    css = open(os.path.join(STATIC, "css", "style.css"), encoding="utf-8").read()
    check(".pkey" in css, "payload field labels are styled")
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

    cpu = _camp("2D-seq", "pi", "M", "20260728-000000", 5, gpu="nvidia-smi not found")
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

    check(curation.normalize_scope(None) == "curated"
          and curation.normalize_scope("all") == "all"
          and curation.normalize_scope("nonsense") == "curated",
          "scope parameter defaults to curated")


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
