#!/usr/bin/env python3
"""One vocabulary for *why a round failed*, shared by the pipeline and trace-vis.

Before this module the answer to "why did this round fail?" was the name of the
first failed check stage, and for 846 of the 1794 rounds here that name was
``build`` — true but useless. Most of those rounds never got as far as a
compiler: the agent hit a context limit, blew its step cap, or was killed by
walltime, so no ``solution.json`` was ever written and the build stage failed by
default. "build" hid all of that behind one word.

The reason a round failed therefore comes from two independent places:

  * **How the agent's run ended** — the terminal record in its trajectory
    (``context_overflow``, ``format_error``, ``max_turns``, ...). Extracted per
    agent schema by the ``*_end_reason`` functions below and recorded in
    ``.eval/stats.json`` by ``checks/trace_stats.py``.
  * **What the harness found afterwards** — no artifact, an artifact that will
    not compile, a run that crashed. Read off ``facts.json`` / ``build.log`` /
    ``run.log`` by :func:`classify_artifact`.

:func:`fail_class` combines them into ONE slug per round with a fixed
precedence, and ``checks/collect_results.py`` writes it to every row of
``results/runs.jsonl`` as ``fail_class``. Pass/fail itself is untouched: this
module never decides whether a round passed, only how to name the failure, so
every pass@1 number is exactly what it was before.

Pure stdlib and side-effect free so both consumers can import it: the host
pipeline (``checks/``) and the trace viewer, which gets a copy baked into its
image at ``/opt/trace-vis/failure_class.py``.
"""

from __future__ import annotations

import json
import os
import re

# ---------------------------------------------------------------------------
# Run-end reasons: why the AGENT stopped.
#
# The provider/harness message is the only place the real cause is written
# down. Without parsing it a truncated run just stops mid-conversation and looks
# like the model gave up.
#
# Each entry: (regex over the provider message, reason slug, (used group, limit
# group), what the "used" number counts). Groups are 1-based; None means that
# number is not in the message. The noun matters: some providers report the
# input length, others the input+completion total, and calling a total "input
# tokens" would misstate what overflowed.
# ---------------------------------------------------------------------------
END_PATTERNS = (
    # sglang / vLLM behind LiteLLM ("Input length (N tokens) exceeds the maximum
    # allowed length (M tokens)")
    (r"Input length \((\d+) tokens\) exceeds the maximum allowed length \((\d+) tokens\)",
     "context_overflow", (1, 2), "input tokens"),
    # provider-side context check: total = input + requested completion
    (r"maximum context length of (\d+) tokens\. You requested a total of (\d+) tokens",
     "context_overflow", (2, 1), "tokens in total (input + requested completion)"),
    # vLLM's own wording, which litellm forwards verbatim inside a
    # ContextWindowExceededError ("maximum context length is N tokens. However,
    # you requested ... for a total of at least M tokens")
    (r"maximum context length is (\d+) tokens.*?total of at least (\d+) tokens",
     "context_overflow", (2, 1), "tokens in total (input + requested completion)"),
    # litellm's ContextWindowExceededError as swe-mini records it in `role:"exit"`
    (r"The input \((\d+) tokens\) is longer than the model's context length \((\d+) tokens\)",
     "context_overflow", (1, 2), "input tokens"),
    (r"max_completion_tokens is too large: (\d+)\.\s*This model supports at most (\d+)",
     "output_limit", (1, 2), "requested output tokens"),
    (r"(?:context length|context window|too many tokens|maximum context)",
     "context_overflow", (None, None), None),
    (r"(?:rate.?limit|throttl|\b429\b)", "rate_limit", (None, None), None),
    (r"(?:timed out|timeout)", "timeout", (None, None), None),
)


def scan_end_message(msg, extra=()):
    """(reason, used, limit, noun) for a provider error string, or Nones."""
    for pat, reason, (gu, gl), what in tuple(extra) + END_PATTERNS:
        m = re.search(pat, msg or "", re.I)
        if not m:
            continue
        return (reason, int(m.group(gu)) if gu else None,
                int(m.group(gl)) if gl else None, what)
    return None, None, None, None


def token_phrase(used, limit, what):
    """'N <noun> against an M limit (K over)' — the shared way to state a cap."""
    return (f"{used:,} {what or 'tokens'} against a {limit:,} limit "
            f"({used - limit:,} over)")


END_LABEL = {
    "completed": "Completed — the model ended its turn",
    "context_overflow": "Context window exceeded",
    "output_limit": "Output token limit reached",
    "rate_limit": "Rate limited by the provider",
    "timeout": "Request timed out",
    "api_error": "Provider error",
    "tool_use": "Stopped on a tool call",
    "cut_off": "Cut off — no final assistant message",
    "unknown": "Ended for an unrecorded reason",
    # kimi-only reasons (see kimi_terminal_reason) — pi never produces these
    "max_steps_exceeded": "Turn step cap reached",
    "connection_error": "Lost connection to the provider",
    "empty_response": "Provider returned no text or tool call",
    "hung_after_final": "Hung after its final reply — the harness ended it",
    # claude-only reasons (see claude_end_reason)
    "max_turns": "Turn limit reached — the harness cut the run off",
    "thinking_block": "Provider rejected the assistant's thinking block",
    # swe-mini-only reasons (see swemini_end_reason)
    "limits_exceeded": "Step or cost limit reached — the harness stopped the run",
    "format_error": "Too many malformed tool calls in a row — the harness gave up",
}

END_SEVERITY = {"completed": "ok", "tool_use": "warn", "cut_off": "warn",
                "unknown": "warn", "output_limit": "warn",
                "max_steps_exceeded": "warn", "max_turns": "warn",
                "limits_exceeded": "warn", "format_error": "warn"}


def provider_message(err):
    """Pull the human part out of pi's `errorMessage` ('500: {"message": ...}')."""
    if not isinstance(err, str):
        return ""
    body = err.split(": ", 1)[1] if ": " in err[:8] and err[:3].isdigit() else err
    body = body.strip()
    if body.startswith("{"):
        try:
            d = json.loads(body)
            if isinstance(d, dict) and isinstance(d.get("message"), str):
                return d["message"]
        except ValueError:
            pass
    return err


def pi_end_reason(stop, err):
    """(reason slug, tokens used, token limit, noun) for a pi run's final message."""
    if stop == "stop":
        return "completed", None, None, None
    if stop == "length":
        return "output_limit", None, None, None
    if stop == "toolUse":
        return "tool_use", None, None, None
    if stop is None:
        return "cut_off", None, None, None
    if stop != "error":
        return "unknown", None, None, None

    reason, used, limit, what = scan_end_message(provider_message(err))
    if reason is None:
        return "api_error", None, None, None
    return reason, used, limit, what


CLAUDE_END_PATTERNS = (
    # Claude Code's own phrasings, which never reach pi/kimi
    (r"prompt is too long", "context_overflow", (None, None), None),
    (r"hit your session limit", "rate_limit", (None, None), None),
    (r"content block is not a thinking block", "thinking_block", (None, None), None),
    (r"connection closed|cannot connect to host|connection error",
     "connection_error", (None, None), None),
)


def claude_end_reason(res):
    """(reason slug, tokens used, token limit, noun) for a claude result record."""
    if res is None:
        return "cut_off", None, None, None
    term = res.get("terminal_reason")
    if term == "max_turns" or res.get("subtype") == "error_max_turns":
        return "max_turns", None, None, None
    if res.get("is_error") or term in ("api_error", "blocking_limit"):
        # the gateway errors are the same litellm/vLLM messages pi and kimi see,
        # so the shared patterns apply after claude's own wordings
        reason, used, limit, what = scan_end_message(res.get("result") or "",
                                                     CLAUDE_END_PATTERNS)
        return (reason or "api_error", used, limit, what)
    if res.get("stop_reason") == "max_tokens":
        return "output_limit", None, None, None
    if term == "completed" or res.get("subtype") == "success":
        return "completed", None, None, None
    return "unknown", None, None, None


# keyed lowercase: the status is a class name and its casing has changed
# between mini-swe-agent versions
SWEMINI_EXIT_REASON = {
    "submitted": "completed",
    "limitsexceeded": "limits_exceeded",
    "repeatedformaterror": "format_error",
}


# swe-mini calls a run that ended in malformed tool calls a RepeatedFormatError,
# which names the SYMPTOM. The deeper cause is often a model with no room left
# to answer in: the provider truncates the response (``finish_reason: length``)
# before it can close a tool call, the harness sees no tool call at all, and
# after ``max_consecutive_format_errors`` of those it gives up. The receipts are
# on the FormatError interrupt records the harness wrote itself.
FORMAT_ERROR_INTERRUPT = "FormatError"

# When the truncated answer got less than this share of the tokens in play, the
# PROMPT is what ate the budget: the conversation filled the model's context
# window and left no room to reply. Above it, the reply ran into an output cap.
FORMAT_CONTEXT_SHARE = 0.1


def _trailing_format_errors(messages):
    """The unbroken run of FormatError interrupts that ends a trajectory."""
    trail = []
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            break
        if msg.get("role") == "exit":
            continue
        extra = msg.get("extra")
        extra = extra if isinstance(extra, dict) else {}
        if extra.get("interrupt_type") != FORMAT_ERROR_INTERRUPT:
            break
        trail.append(msg)
    return trail


def _interrupt_usage(msg):
    """(finish_reason, prompt tokens, completion tokens) of the offending call."""
    extra = msg.get("extra")
    resp = (extra.get("response") if isinstance(extra, dict) else None) or {}
    choice = (resp.get("choices") or [{}])[0] or {}
    usage = resp.get("usage") or {}
    return (choice.get("finish_reason"),
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0))


def swemini_format_cause(messages):
    """(reason, note) for what REALLY ended a RepeatedFormatError run, or None.

    ``None`` means the format errors were genuine: the model answered and simply
    never produced a well-formed tool call, so ``format_error`` stands.
    """
    trail = _trailing_format_errors(messages)
    if not trail:
        return None
    seen = [_interrupt_usage(m) for m in trail]
    if not all(finish == "length" for finish, _, _ in seen):
        return None

    prompt = max(p for _, p, _ in seen)
    answer = max(c for _, _, c in seen)
    note = f"cut off mid-answer {len(trail)}x in a row (finish_reason: length)"
    if prompt or answer:
        note += f", {answer:,} tokens of reply after a {prompt:,}-token prompt"
    if (prompt + answer) and answer <= (prompt + answer) * FORMAT_CONTEXT_SHARE:
        return "context_overflow", note + " — the window was full, not a formatting problem"
    return "output_limit", note


def swemini_end_reason(status, msg, messages=()):
    """(reason slug, used, limit, noun, note) for a swe-mini ``role:"exit"``.

    ``status`` is the harness's own exception class name; ``msg`` its message.
    An absent status means the trajectory has no exit record at all — the
    process was killed before it could write one. ``messages`` is the whole
    trajectory, needed only to look behind a RepeatedFormatError; ``note`` is
    the plain-words finding when that lookup changed the answer, else ``""``.
    """
    if status is None and not msg:
        return "cut_off", None, None, None, ""
    reason = SWEMINI_EXIT_REASON.get(str(status).lower()) if status else None
    if reason == "format_error":
        cause = swemini_format_cause(messages)
        if cause:
            return cause[0], None, None, None, cause[1]
    if reason is not None:
        return reason, None, None, None, ""
    reason, used, limit, what = scan_end_message(msg or "")
    return (reason or ("api_error" if status else "unknown"), used, limit, what, "")


KIMI_STDERR_TAIL = 8000  # the terminal line is always within the last few KB

KIMI_TERMINAL_RE = re.compile(
    r"error: failed to run prompt:\s*([\w.]+)(?::\s*(.*?))?\s*(?:\r?\nSee log:|\Z)",
    re.DOTALL)
KIMI_MAX_STEPS_RE = re.compile(r"maxSteps=(\d+)")
KIMI_EMPTY_RESPONSE_RE = re.compile(r"only thinking content|empty response", re.I)


def kimi_terminal_reason(tail):
    """(reason, message, {used, limit, what}) from a kimi stderr tail.

    ``None`` when the tail holds no terminal ``error: failed to run prompt:``
    line — either the run completed or stderr was not readable.
    """
    matches = list(KIMI_TERMINAL_RE.finditer(tail or ""))
    if not matches:
        return None
    domain_kind, msg = matches[-1].groups()
    msg = (msg or "").strip()
    kind = domain_kind.rsplit(".", 1)[-1]

    if kind in ("max_steps_exceeded", "connection_error", "rate_limit"):
        return kind, msg, {}
    if domain_kind == "context.overflow":
        # server context 400 with compaction off ("Compaction limit exceeded (0)")
        return "context_overflow", msg, {}
    if kind == "api_error":
        # kimi's provider errors wrap the same litellm/vLLM messages pi sees,
        # so the same context/output-limit patterns apply verbatim.
        reason, used, limit, what = scan_end_message(msg)
        if reason is not None:
            return reason, msg, {"used": used, "limit": limit, "what": what}
        if KIMI_EMPTY_RESPONSE_RE.search(msg):
            return "empty_response", msg, {}
        return "api_error", msg, {}
    return kind or "unknown", msg, {}


def read_kimi_stderr_tail(round_dir):
    """Last few KB of a kimi round's ``agent.stderr.log``; '' if unreadable."""
    try:
        path = os.path.join(round_dir, "agent.stderr.log")
        size = os.path.getsize(path)
        with open(path, "r", errors="replace") as fh:
            if size > KIMI_STDERR_TAIL:
                fh.seek(size - KIMI_STDERR_TAIL)
            return fh.read()
    except OSError:
        return ""


# kimi's tool result for a command it moved to the background.
KIMI_BACKGROUND_TASK = "The task now runs in the background"


def kimi_hung(round_dir, final_has_toolcalls, backgrounded):
    """Detail line when kimi stayed alive after its final no-tool reply, else "".

    A background task keeps kimi running after it has answered. Since
    2026-09-25 kimi solve.sh ends it after 10 min and writes
    ``agent.end_watch.log`` with ``hung_after_final``. Older rounds hit the
    same thing without the watch and ran into the 4h limit: ``.solve_timeout``,
    a final reply with no tool call, and a backgrounded command."""
    try:
        with open(os.path.join(round_dir, "agent.end_watch.log"), errors="replace") as fh:
            for line in fh:
                if "hung_after_final" in line:
                    return line.strip()
    except OSError:
        pass
    if (os.path.exists(os.path.join(round_dir, ".solve_timeout")) and not final_has_toolcalls
            and backgrounded):
        return "solve timeout after the final reply, with a background task running"
    return ""


def read_kimi_wire(round_dir):
    """Per-call ``(context, finishReason, output)`` from the main-agent wire log,
    plus the step count before each ``compaction.started`` record.

    kimi solve.sh copies each session's ``wire.jsonl`` to
    ``agent.kimi-wire/<agent>.wire.jsonl``; ``main`` is the top-level agent.
    One ``step.end`` loop event per model call; context is the full prompt
    (``inputOther + inputCacheRead + inputCacheCreation``). ``None`` when the
    round has no wire log (every kimi round before 2026-09-17)."""
    wire = os.path.join(round_dir, "agent.kimi-wire", "main.wire.jsonl")
    if not os.path.isfile(wire):
        return None
    steps, points = [], []
    with open(wire, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            ev = rec.get("event") if isinstance(rec.get("event"), dict) else {}
            if "compaction.started" in (rec.get("type"), ev.get("type")):
                points.append(len(steps))
            if rec.get("type") != "context.append_loop_event":
                continue
            if ev.get("type") != "step.end":
                continue
            usage = ev.get("usage") or {}
            context = sum(usage.get(k) or 0 for k in
                          ("inputOther", "inputCacheRead", "inputCacheCreation"))
            steps.append((context, ev.get("finishReason"), usage.get("output") or 0))
    return steps, points


def kimi_end_reason(round_dir, last_role, last_has_toolcalls, last_meta_type,
                    final_has_toolcalls=False, backgrounded=False, wire_steps=None):
    """(reason, detail, used, limit, what): why a kimi round stopped.

    ``last_*`` describe the stream's last line, ``final_has_toolcalls`` the last
    assistant reply, ``backgrounded`` whether a command went to the background,
    ``wire_steps`` the first item of :func:`read_kimi_wire` (None without it).
    Order: a stderr terminal reason, then the output cap from the wire log (the
    stream alone looks completed), then a hang after the final reply, then the
    shape of the stream end."""
    stderr = kimi_terminal_reason(read_kimi_stderr_tail(round_dir))
    if stderr is not None:
        reason, msg, extra = stderr
        return reason, msg, extra.get("used"), extra.get("limit"), extra.get("what")
    if wire_steps and wire_steps[-1][1] == "max_tokens":
        return "output_limit", "", wire_steps[-1][2], None, "output tokens"
    hung = kimi_hung(round_dir, final_has_toolcalls, backgrounded)
    if hung:
        return "hung_after_final", hung, None, None, None
    if last_role == "assistant" and not last_has_toolcalls:
        reason = "completed"
    elif last_role == "meta" and last_meta_type == "session.resume_hint":
        reason = "completed"
    elif last_role == "tool" or (last_role == "assistant" and last_has_toolcalls):
        reason = "cut_off"
    else:
        reason = "unknown"
    return reason, "", None, None, None


# ---------------------------------------------------------------------------
# Artifact reasons: what the HARNESS found after the agent stopped.
# ---------------------------------------------------------------------------

# measure.sh writes a prose failure_reason when it refuses to replay anything.
# Map its exact strings to slugs; the slug is what everything downstream keys
# on. Newer runs also carry `failure_code` directly (see agents/measure.sh).
FACTS_FAILURE_CODE = {
    "No solution.json produced": "no_solution",
    "Incomplete solution.json (missing build/run)": "incomplete_solution",
}

# First match wins, so the order is the classification. Link errors are checked
# before compile errors because a failing link also prints "error:" (from
# collect2/ld), and the make-level stops come first because they abort before
# any compiler ran at all.
BUILD_PATTERNS = (
    # the problem Makefiles bootstrap a config.mk and stop on purpose; an agent
    # that never re-runs make ends the round here having compiled nothing
    (r"Stopping after creating config\.mk", "make_config_stop"),
    (r"make(?:\[\d+\])?: (\S+): (?:Command not found|No such file or directory)",
     "toolchain_missing"),
    (r"No rule to make target", "make_no_rule"),
    (r"undefined reference to|collect2: error: ld returned|ld: cannot find",
     "link_error"),
    (r"ptxas .*(?:error|fatal)", "ptxas_error"),
    (r"error: |errors? detected in the compilation|fatal error:", "compile_error"),
    (r"No such file or directory", "missing_file"),
)

# The graded run's own error line. Unlike a compiler, a runner often prints a
# usage banner AFTER the thing that went wrong, so the last line of run.log is
# routinely the wrong quote ("Build with: make TOOLCHAIN=CLANG"). Take the first
# line that looks like a diagnosis instead, and fall back to the last line.
RUN_ERROR_PATTERN = (r"\berror\b|\bfatal\b|segmentation fault|bus error|core dumped|"
                     r"terminate called|assertion|traceback \(most recent call last\)|"
                     r"cuda (?:error|failure)|out of memory|killed|"
                     r"not found|no such file|permission denied|"
                     r"(?:illegal|invalid|unrecognized|unknown) option|command not found|"
                     r"\bunknown\b|\bunsupported\b|\bfailed\b|\bcannot\b|"
                     r"\bfopen\b|^usage:")

_BUILD_LOG_HEAD = 512 * 1024   # compiler errors appear anywhere in the log, so
_BUILD_LOG_TAIL = 64 * 1024    # scan both ends of a pathological one


def _read_log(path, head=_BUILD_LOG_HEAD, tail=_BUILD_LOG_TAIL):
    """Log text, head+tail only for a huge one. '' when absent/unreadable."""
    try:
        size = os.path.getsize(path)
        with open(path, "r", errors="replace") as fh:
            if size <= head + tail:
                return fh.read()
            start = fh.read(head)
            fh.seek(size - tail)
            return start + "\n...\n" + fh.read()
    except OSError:
        return ""


def _first_match_line(text, pattern):
    """The first line of ``text`` matching ``pattern`` — evidence for the slug."""
    for line in (text or "").splitlines():
        if re.search(pattern, line, re.I):
            return line.strip()
    return ""


def classify_build_log(text):
    """(slug, evidence line) for a build.log body, or (None, '') if it is clean."""
    for pattern, slug in BUILD_PATTERNS:
        line = _first_match_line(text, pattern)
        if line:
            return slug, line
    return None, ""


def _load(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


# timeout(1)'s exit code for "the command outlived its cap" — measure.sh caps
# the build at 20 min and the graded run at 10 min.
TIMEOUT_STATUS = 124


def classify_artifact(eval_dir, facts=None):
    """(slug, detail) for what the harness found in ``eval_dir``, or (None, '').

    Answers "where did it fail?" for the build/run half of a round: no artifact
    at all, an artifact that will not compile, or a graded run that crashed.
    Never opens a compiler — it only reads the receipts already captured.
    """
    eval_dir = str(eval_dir)
    facts = facts if facts is not None else _load(os.path.join(eval_dir, "facts.json"))

    code = facts.get("failure_code") or FACTS_FAILURE_CODE.get(facts.get("failure_reason"))
    if code:
        return code, facts.get("failure_reason") or ""
    if facts.get("failure_reason"):
        # an unmapped prose reason from a future measure.sh — surface it rather
        # than silently calling it something else
        return "measure_refused", facts["failure_reason"]

    build = facts.get("build") or {}
    run = facts.get("run") or {}
    bstatus, rstatus = build.get("status"), run.get("status")

    if bstatus not in (0, None):
        if bstatus == TIMEOUT_STATUS:
            return "build_timeout", "build hit measure.sh's 20 minute cap"
        slug, line = classify_build_log(_read_log(os.path.join(eval_dir, "build.log")))
        if slug:
            return slug, line
        return "build_error", f"build exited {bstatus}, log has no recognised error"

    if rstatus not in (0, None):
        if rstatus == TIMEOUT_STATUS:
            return "run_timeout", "graded run hit measure.sh's 10 minute cap"
        text = _read_log(os.path.join(eval_dir, "run.log"))
        line = _first_match_line(text, RUN_ERROR_PATTERN)
        if not line:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            line = lines[-1] if lines else ""
        return "run_error", (line or f"run exited {rstatus}")

    return None, ""


# ---------------------------------------------------------------------------
# The combined answer.
# ---------------------------------------------------------------------------

# Stages that grade a run that actually happened. When one of these fails there
# IS a measurement, and the measurement is the reason — how the agent's session
# ended is secondary.
GRADED_STAGES = ("mse", "metric", "perf", "gpu", "digest", "integrity")

FAIL_CLASS_LABEL = {
    # artifact
    "no_solution": "No solution.json — the agent never wrote the artifact",
    "incomplete_solution": "solution.json has no build/run command",
    "measure_refused": "The harness refused to replay this round",
    "make_config_stop": "Make stopped after writing config.mk and was never re-run",
    "toolchain_missing": "A build tool is missing from the image",
    "make_no_rule": "Make has no rule for a required target",
    "link_error": "Link error",
    "ptxas_error": "The CUDA assembler (ptxas) rejected the kernel",
    "compile_error": "Compile error",
    "missing_file": "The build referenced a file that is not there",
    "build_timeout": "The build outlived its 20 minute cap",
    "build_error": "The build failed for an unrecognised reason",
    "run_timeout": "The graded run outlived its 10 minute cap",
    "run_error": "The graded run crashed",
    # graded stages
    "mse": "Wrong numbers — error above the correctness threshold",
    "metric": "No usable metric was produced",
    "perf": "Too slow against the reference kernel",
    "gpu": "The GPU was idle — the work did not run on it",
    "digest": "Output digest does not match the reference",
    "integrity": "A frozen file was modified",
    "unknown": "Failed for an unrecorded reason",
}
FAIL_CLASS_LABEL.update({k: v for k, v in END_LABEL.items() if k != "completed"})


def failed_stage(stages):
    """Name of the first stage that actually failed (skipped ones don't count)."""
    for st in stages or []:
        if not st.get("passed") and not st.get("skipped"):
            return st.get("name")
    return None


def fail_class(passed, stages=(), end_reason=None, eval_dir=None, facts=None):
    """(slug, detail) naming why one round failed — ``(None, '')`` when it passed.

    Precedence, in words:

    1. A failed GRADED stage wins. The round built, ran, and produced numbers;
       those numbers are the reason, whatever the agent's session did next.
    2. Otherwise take the harness's artifact finding (:func:`classify_artifact`).
       If that finding is "there is no artifact" AND the agent's run ended
       badly, the run-end reason wins instead — a missing solution.json is the
       symptom, `context_overflow` / `format_error` / `max_turns` is the cause.
    3. Otherwise fall back to the run-end reason, then to the stage name.

    The old answer to this question was the bare stage name, which was "build"
    for nearly half of all rounds and said nothing about where the run actually
    broke. ``build`` is never returned by this function.
    """
    if passed:
        return None, ""

    stage = failed_stage(stages)
    if stage in GRADED_STAGES:
        return stage, _stage_detail(stages, stage)

    slug, detail = classify_artifact(eval_dir, facts) if (eval_dir or facts) else (None, "")
    ended_badly = bool(end_reason) and end_reason != "completed"

    if slug in ("no_solution", "incomplete_solution", "measure_refused") and ended_badly:
        return end_reason, END_LABEL.get(end_reason, end_reason)
    if slug:
        return slug, detail
    if ended_badly:
        return end_reason, END_LABEL.get(end_reason, end_reason)
    # The receipts contradict the verdict (the stage failed on something other
    # than an exit status). Keep the stage, but in the fine-grained vocabulary
    # so no row ever reads bare "build".
    if stage in ("build", "run"):
        return f"{stage}_error", _stage_detail(stages, stage)
    return stage or "unknown", _stage_detail(stages, stage)


def _stage_detail(stages, name):
    for st in stages or []:
        if st.get("name") == name:
            return st.get("detail") or ""
    return ""
