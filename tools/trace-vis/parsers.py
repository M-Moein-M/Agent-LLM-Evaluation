"""Normalizes the 4 distinct agent trajectory schemas into one timeline shape.

Schemas in this repo (see AGENT.md):
  pi          agent.jsonl        - line-delimited envelope events (session/
                                    message_start/message_end/tool_execution_*)
  swe-mini    agent.traj.json    - single JSON doc, OpenAI chat `messages` list
  claude(-pro) agent.stream.jsonl - line-delimited Anthropic `/v1/messages`
                                    stream records (type: system/assistant/user/result)
  kimi        agent.stream.jsonl - line-delimited RAW OpenAI chat messages,
                                    no envelope "type" field at all

All four normalize to: {"events": [...], "session_meta": {...}, "warnings": [...]}
Event kinds: system | user | assistant_text | thinking | tool_call | log | run_end

An assistant turn carrying tool calls but no prose emits NO assistant_text
event -- it is represented entirely by its tool_call events. So counting
assistant_text events undercounts model calls (kimi in particular answers
with a bare tool call about 30% of the time); the model-call count lives in
stats.json's ``turns``, not in this event list.
(keep this list in sync with FILTER_KINDS/LABEL in static/js/timeline.js)

Most events carry finish_reason=None. It's set only when THAT SPECIFIC turn's
own stop/finish reason was abnormal (e.g. litellm's "length", pi's "error") —
separate from the terminal run_end event, and can appear mid-timeline, e.g. a
turn the harness silently retried after. pi sets it directly from each
message's own stopReason; swe-mini's FormatError turns never become a real
assistant message at all (see _swemini_format_error_turn), so one is
synthesized from the hidden raw response attached to the correction message.
claude's per-turn stop_reason isn't populated on the stream records (only the
final `result` has one, already covered by run_end) and kimi's raw stream
carries no finish_reason of any kind, so neither parser sets this field.
"""
import json
import os
import re
import sys

# The failure vocabulary (why a run stopped, what the harness found) is shared
# with the eval pipeline so the viewer and the statistics in results/ can never
# drift apart: one edit to checks/failure_class.py changes both. The image bakes
# a copy in next to this file; a run from the source tree finds it in checks/.
try:
    import failure_class as fc
except ImportError:
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "checks"))
    import failure_class as fc

_scan_end_message = fc.scan_end_message
_token_phrase = fc.token_phrase
_END_LABEL = fc.END_LABEL
_END_SEVERITY = fc.END_SEVERITY
_provider_message = fc.provider_message
_pi_end_reason = fc.pi_end_reason
_claude_end_reason = fc.claude_end_reason
_swemini_end_reason = fc.swemini_end_reason

MAX_TEXT = 20000    # cap per-field text sent to the browser; UI shows a truncation note
MAX_EVENTS = 4000   # hard cap on timeline nodes (largest trace here is 75 MB)


def _cap(s):
    if s is None:
        return s, False
    if not isinstance(s, str):
        try:
            s = json.dumps(s, indent=2, default=str)
        except Exception:
            s = str(s)
    if len(s) > MAX_TEXT:
        return s[:MAX_TEXT], True
    return s, False


def _norm_ts(ts):
    if ts is None:
        return None
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return None
    return ts / 1000.0 if ts > 1e12 else ts


def _new_event(kind, **kw):
    ev = {"kind": kind, "role": None, "text": None, "text_truncated": False,
          "tool_name": None, "tool_call_id": None, "tool_input": None,
          "tool_output": None, "tool_output_truncated": False,
          "is_error": False, "timestamp": None,
          # set only when THIS turn's own stop/finish reason was abnormal —
          # distinct from end_reason, which is the terminal event for the
          # whole run. None here means "normal turn", not "unknown".
          "finish_reason": None, "finish_reason_detail": None}
    ev.update(kw)
    return ev


def detect_kind(round_dir):
    """Return (kind, filename) for the trace file present in a round dir, or (None, None)."""
    for fname, kind in (("agent.jsonl", "pi"),
                        ("agent.traj.json", "swe-mini"),
                        ("agent.stream.jsonl", "stream")):
        if os.path.isfile(os.path.join(round_dir, fname)):
            if kind == "stream":
                # Disambiguate claude/claude-pro vs kimi by peeking the first line.
                # Can't key off "type" alone: kimi's own `role: meta` records (e.g.
                # a startup retry logged before any real message) carry a "type"
                # field too ({"role":"meta","type":"turn.step.retrying",...}),
                # which used to misdetect those traces as claude and silently
                # parse to zero events. claude records never carry a top-level
                # "role" key (it's nested under "message"), so that's the
                # reliable signal instead.
                try:
                    with open(os.path.join(round_dir, fname), "r", errors="replace") as f:
                        first = f.readline()
                    if first.strip():
                        d = json.loads(first)
                        kind = "kimi" if "role" in d else "claude"
                    else:
                        # agent.stream.jsonl is empty: kimi failed before writing
                        # any output (e.g. a rate limit on the very first request).
                        # Fall back to agent.stderr.log — its terminal line
                        # ("error: failed to run prompt: <domain>.<kind>: ...")
                        # is kimi CLI's own error format, never produced by the
                        # other three harnesses.
                        kind = "kimi" if fc.kimi_terminal_reason(
                            fc.read_kimi_stderr_tail(round_dir)) else "claude"
                except Exception:
                    kind = "claude"
            return kind, fname
    return None, None


def parse_trace(round_dir):
    """Parse the round's trajectory file into a normalized, schema-independent form.

    Returns the FULL event list; callers paginate via `slice_events`. Results are
    worth caching by (path, mtime) — the largest trace in this repo is 75 MB.
    """
    kind, fname = detect_kind(round_dir)
    if kind is None:
        return {"events": [], "session_meta": {}, "warnings": ["no recognized trace file in round dir"],
                "schema": None, "source_file": None, "stats": _stats([])}
    path = os.path.join(round_dir, fname)
    try:
        if kind == "pi":
            result = _parse_pi(path)
        elif kind == "swe-mini":
            result = _parse_swemini(path)
        elif kind == "claude":
            result = _parse_claude(path)
        elif kind == "kimi":
            result = _parse_kimi(path)
        else:
            result = {"events": [], "session_meta": {}, "warnings": [f"unknown kind {kind}"]}
    except Exception as e:
        result = {"events": [], "session_meta": {}, "warnings": [f"parse error: {e!r}"]}

    events = result.get("events", [])
    for i, ev in enumerate(events):
        ev["idx"] = i
    result["schema"] = kind
    result["source_file"] = fname
    result["stats"] = _stats(events)
    try:
        result["stats"]["file_bytes"] = os.path.getsize(path)
    except OSError:
        pass
    return result


def _stats(events):
    counts = {}
    tools = {}
    errors = 0
    for ev in events:
        counts[ev["kind"]] = counts.get(ev["kind"], 0) + 1
        if ev["kind"] == "tool_call":
            name = ev.get("tool_name") or "?"
            tools[name] = tools.get(name, 0) + 1
            if ev.get("is_error"):
                errors += 1
    return {"total_events": len(events), "by_kind": counts,
            "by_tool": tools, "tool_errors": errors}


def slice_events(result, offset=0, limit=MAX_EVENTS, kinds=None):
    """Paginate/filter a parsed result for transport to the browser."""
    events = result.get("events", [])
    if kinds:
        events = [e for e in events if e["kind"] in kinds]
    total = len(events)
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), MAX_EVENTS))
    page = events[offset:offset + limit]
    return {
        "schema": result.get("schema"),
        "source_file": result.get("source_file"),
        "session_meta": result.get("session_meta", {}),
        "warnings": result.get("warnings", []),
        "stats": result.get("stats", {}),
        "events": page,
        "offset": offset,
        "limit": limit,
        "returned": len(page),
        "total_filtered": total,
        "has_more": offset + len(page) < total,
    }


def _iter_jsonl(path):
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------- pi ---

# pi records why a run stopped in the trailing `agent_end` envelope; a
# context-window overflow arrives as a provider error string buried two JSON
# levels deep. fc.pi_end_reason() turns that into a slug; this only phrases it.
def _pi_run_end(end, retry):
    """Terminal timeline event: why this pi run stopped. None if pi never said."""
    if end is None:
        return None
    reason, used, limit, what = _pi_end_reason(end.get("stop"), end.get("error"))
    parts = [_END_LABEL.get(reason, reason)]
    if used and limit:
        parts.append(_token_phrase(used, limit, what))
    elif reason == "output_limit":
        parts.append("the reply stopped at the model's output token cap")
    elif reason == "cut_off":
        parts.append("the run stops after the last tool result — turn limit, "
                     "walltime, or the harness stopped it")
    if retry and not retry.get("success"):
        parts.append(f"retried {retry.get('attempt')} times, all failed")
    elif end.get("will_retry"):
        parts.append("the agent retried after this")

    detail = _provider_message(end.get("error")) or end.get("error") or ""
    text, trunc = _cap(" · ".join(parts))
    detail_text, _ = _cap(detail)
    return _new_event(
        "run_end", role="system", text=text, text_truncated=trunc,
        tool_output=detail_text or None,
        is_error=_END_SEVERITY.get(reason, "error") == "error",
        timestamp=end.get("ts"),
        end_reason=reason,
        end_severity=_END_SEVERITY.get(reason, "error"),
        end_stop_reason=end.get("stop"),
        end_tokens_used=used, end_tokens_limit=limit,
    )


def _parse_pi(path):
    events = []
    warnings = []
    meta = {}
    pending_calls = {}  # toolCallId -> event dict (index into events)
    end = None          # last agent_end: pi can emit one per retry attempt
    retry = None        # last auto_retry_end, which lands *after* agent_end

    for d in _iter_jsonl(path):
        t = d.get("type")
        if t == "agent_end":
            msgs = d.get("messages") or []
            last = msgs[-1] if msgs else {}
            end = {"stop": last.get("stopReason"), "error": last.get("errorMessage"),
                   "will_retry": bool(d.get("willRetry")),
                   "ts": _norm_ts(last.get("timestamp"))}
        elif t == "auto_retry_end":
            retry = {"success": bool(d.get("success")), "attempt": d.get("attempt"),
                     "error": d.get("finalError")}
        elif t == "session":
            meta.update({"session_id": d.get("id"), "cwd": d.get("cwd"), "started": d.get("timestamp")})
        elif t == "message_end":
            msg = d.get("message", {}) or {}
            role = msg.get("role")
            ts = _norm_ts(msg.get("timestamp"))
            if role == "user":
                for block in msg.get("content", []) or []:
                    if block.get("type") == "text":
                        text, trunc = _cap(block.get("text"))
                        events.append(_new_event("user", role="user", text=text, text_truncated=trunc, timestamp=ts))
            elif role == "assistant":
                meta.setdefault("model", msg.get("model"))
                meta.setdefault("provider", msg.get("provider"))
                # pi records a stopReason on EVERY assistant turn, not just the
                # run's last one (that's agent_end, handled below). A turn cut
                # at "length" or killed by "error" but auto-retried leaves the
                # run looking fine overall — mark it here so it's not invisible.
                stop = msg.get("stopReason")
                abnormal = stop in ("length", "error")
                fr_detail = None
                if abnormal:
                    fr_detail = (_provider_message(msg.get("errorMessage")) if stop == "error"
                                 else "response was cut off at the output token cap")
                blocks = msg.get("content", []) or []
                made_event = False
                for block in blocks:
                    bt = block.get("type")
                    if bt == "text" and block.get("text"):
                        text, trunc = _cap(block.get("text"))
                        events.append(_new_event("assistant_text", role="assistant", text=text, text_truncated=trunc,
                                                  timestamp=ts, finish_reason=stop if abnormal else None,
                                                  finish_reason_detail=fr_detail))
                        made_event = True
                    elif bt == "thinking":
                        text, trunc = _cap(block.get("thinking"))
                        events.append(_new_event("thinking", role="assistant", text=text, text_truncated=trunc,
                                                  timestamp=ts, finish_reason=stop if abnormal else None,
                                                  finish_reason_detail=fr_detail))
                        made_event = True
                    elif bt == "toolCall":
                        ev = _new_event("tool_call", role="assistant", tool_name=block.get("name"),
                                         tool_call_id=block.get("id"), tool_input=block.get("arguments"),
                                         timestamp=ts, finish_reason=stop if abnormal else None,
                                         finish_reason_detail=fr_detail)
                        events.append(ev)
                        if block.get("id"):
                            pending_calls[block["id"]] = ev
                        made_event = True
                if abnormal and not made_event:
                    # the call failed/was cut before producing any usable content
                    text, trunc = _cap(fr_detail or stop)
                    events.append(_new_event("log", role="assistant", text=text, text_truncated=trunc,
                                              timestamp=ts, is_error=(stop == "error"),
                                              finish_reason=stop, finish_reason_detail=fr_detail))
            elif role == "toolResult":
                call_id = msg.get("toolCallId")
                ev = pending_calls.get(call_id)
                out_parts = []
                for block in msg.get("content", []) or []:
                    if block.get("type") == "text":
                        out_parts.append(block.get("text", ""))
                out_text, trunc = _cap("\n".join(out_parts))
                if ev is not None:
                    ev["tool_output"] = out_text
                    ev["tool_output_truncated"] = trunc
                    ev["is_error"] = bool(msg.get("isError"))
                else:
                    events.append(_new_event("log", text=f"orphan tool result for {call_id}",
                                              tool_output=out_text, is_error=bool(msg.get("isError")), timestamp=ts))

    ev = _pi_run_end(end, retry)
    if ev is not None:
        events.append(ev)
    return {"events": events, "session_meta": meta, "warnings": warnings}


# ------------------------------------------------------------ swe-mini ---

# swe-mini writes its ending as a terminal `role:"exit"` message whose
# `extra.exit_status` is a short class name — "Submitted" for a normal finish,
# but also "ContextWindowExceededError", "LimitsExceeded" (the step or cost cap
# in `info.config.agent`) and "RepeatedFormatError" (max_consecutive_format_errors
# malformed tool calls in a row). It used to render as a plain log line reading
# "session exit: <status>", which buried the difference between a run that
# submitted its work and one the provider or the harness killed. A trajectory
# with no exit message at all was killed before it could write one: `cut_off`.
def _swemini_limit_detail(info):
    """Which cap 'LimitsExceeded' actually hit, in words. '' if we can't tell."""
    agent = ((info.get("config") or {}).get("agent") or {}) if info else {}
    stats = (info or {}).get("model_stats") or {}
    calls, steps = stats.get("api_calls"), agent.get("step_limit")
    cost, budget = stats.get("instance_cost"), agent.get("cost_limit")
    if isinstance(calls, int) and isinstance(steps, int) and steps and calls >= steps:
        return f"used all {steps} steps"
    if isinstance(cost, (int, float)) and isinstance(budget, (int, float)) and \
            budget and cost >= budget:
        return f"spent ${cost:.2f} of a ${budget:.2f} budget"
    if isinstance(calls, int) and isinstance(steps, int) and steps:
        return f"{calls} of {steps} steps used"
    return ""


def _swemini_run_end(exit_msg, info, messages=()):
    """Terminal timeline event: why this swe-mini run stopped.

    ``messages`` is the whole trajectory: a RepeatedFormatError is only the
    symptom, and telling a model that cannot format a tool call from one that
    ran out of room to write one takes the interrupt records behind it.
    """
    info = info or {}
    if exit_msg is None:
        return _new_event(
            "run_end", role="system", end_reason="cut_off", end_severity="warn",
            text="Cut off — no final assistant message · the trajectory has no "
                 "exit message, so the process was killed (walltime or a signal) "
                 "before it could write one")
    extra = exit_msg.get("extra") or {}
    status = extra.get("exit_status") if isinstance(extra, dict) else None
    msg = extra.get("exception_str") if isinstance(extra, dict) else None
    msg = msg or exit_msg.get("content") or ""
    reason, used, limit, what, note = _swemini_end_reason(status, msg, messages)

    # "Submitted" is swe-mini's clean finish: the agent ran the submit command
    # itself, which is a stronger statement than "the model ended its turn".
    parts = ["Submitted — the agent finished and handed its work back"
             if reason == "completed" else _END_LABEL.get(reason, reason)]
    if used and limit:
        parts.append(_token_phrase(used, limit, what))
    elif reason == "limits_exceeded":
        detail = _swemini_limit_detail(info)
        if detail:
            parts.append(detail)
    elif reason == "format_error":
        n = ((info.get("config") or {}).get("agent") or {}).get(
            "max_consecutive_format_errors")
        if n:
            parts.append(f"{n} in a row is the configured cap")
    if note:
        parts.append(note)
    # The class name is the harness's own word for the ending; keep it visible
    # even when the label above already says it in prose.
    if status and status not in parts[0]:
        parts.append(f"exit status {status}")

    text, trunc = _cap(" · ".join(parts))
    detail_text, _ = _cap(msg) if msg and msg != status else (None, False)
    return _new_event(
        "run_end", role="system", text=text, text_truncated=trunc,
        tool_output=detail_text or None,
        is_error=_END_SEVERITY.get(reason, "error") == "error",
        end_reason=reason,
        end_severity=_END_SEVERITY.get(reason, "error"),
        end_stop_reason=status,
        end_tokens_used=used, end_tokens_limit=limit,
    )


# mini-swe-agent raises FormatError when a response has no parseable tool call
# (see actions_toolcall.py) and records it as a plain `role:"user"` correction
# message — the model's own (malformed) turn is never added to the trajectory
# as an assistant message at all. The raw completion survives only in this
# message's `extra.response` (attached there purely for cost logging), so
# without reading it the timeline shows a "Tool call error" note from nowhere,
# with no sign of what the model actually said or why it failed to call a tool.
def _swemini_format_error_turn(msg):
    """A synthetic assistant-turn event for a FormatError's hidden raw response,
    or None if this user message isn't one / carries no response to show."""
    extra = msg.get("extra")
    if not isinstance(extra, dict) or extra.get("interrupt_type") != "FormatError":
        return None
    resp = extra.get("response")
    if not isinstance(resp, dict):
        return None
    choices = resp.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    fmsg = choice.get("message") or {}
    finish = choice.get("finish_reason")
    psf = choice.get("provider_specific_fields") or fmsg.get("provider_specific_fields") or {}
    native = psf.get("native_finish_reason") if isinstance(psf, dict) else None
    reasoning = fmsg.get("reasoning_content") or (psf.get("reasoning_content") if isinstance(psf, dict) else None)
    content = fmsg.get("content")
    body = content or reasoning
    if not body:
        return None

    usage = resp.get("usage") or {}
    bits = []
    if usage.get("total_tokens"):
        bits.append(f"{usage.get('prompt_tokens', 0):,} prompt + "
                     f"{usage.get('completion_tokens', 0):,} completion = "
                     f"{usage['total_tokens']:,} total tokens")
    if native and native != finish:
        bits.append(f"native_finish_reason: {native}")
    detail = " · ".join(bits) or None

    text, trunc = _cap(body)
    kind = "assistant_text" if content else "thinking"
    return _new_event(kind, role="assistant", text=text, text_truncated=trunc,
                       is_error=True, finish_reason=finish, finish_reason_detail=detail)


def _parse_swemini(path):
    events = []
    warnings = []
    meta = {}
    try:
        with open(path, "r", errors="replace") as f:
            doc = json.load(f)
    except Exception as e:
        return {"events": [], "session_meta": {}, "warnings": [f"failed to load JSON: {e!r}"]}

    meta["trajectory_format"] = doc.get("trajectory_format")
    info = doc.get("info", {}) or {}
    meta["model_stats"] = info.get("model_stats")
    model_name = None
    cfg = info.get("config", {}) or {}
    try:
        model_name = cfg.get("model", {}).get("model_name")
    except Exception:
        pass
    if model_name:
        meta["model"] = model_name

    pending_calls = {}  # tool_call_id -> event
    exit_msg = None     # absent = the process was killed before it could exit
    messages = doc.get("messages", []) or []

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue  # boilerplate agent scaffolding, not useful in the timeline
        if role == "user":
            ev = _swemini_format_error_turn(msg)
            if ev is not None:
                events.append(ev)
            text, trunc = _cap(msg.get("content"))
            events.append(_new_event("user", role="user", text=text, text_truncated=trunc))
        elif role == "assistant":
            content = msg.get("content")
            if content:
                text, trunc = _cap(content)
                events.append(_new_event("assistant_text", role="assistant", text=text, text_truncated=trunc))
            for call in msg.get("tool_calls") or []:
                fn = call.get("function", {}) or {}
                args = fn.get("arguments")
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except Exception:
                    pass
                ev = _new_event("tool_call", role="assistant", tool_name=fn.get("name"),
                                 tool_call_id=call.get("id"), tool_input=args)
                events.append(ev)
                if call.get("id"):
                    pending_calls[call["id"]] = ev
        elif role == "tool":
            call_id = msg.get("tool_call_id")
            ev = pending_calls.get(call_id)
            out_text, trunc = _cap(msg.get("content"))
            extra = msg.get("extra") or {}
            ts = _norm_ts(extra.get("timestamp")) if isinstance(extra, dict) else None
            is_err = bool(isinstance(extra, dict) and extra.get("returncode") not in (0, None))
            if ev is not None:
                ev["tool_output"] = out_text
                ev["tool_output_truncated"] = trunc
                ev["is_error"] = is_err
                ev["timestamp"] = ts
            else:
                events.append(_new_event("log", text=f"orphan tool result for {call_id}",
                                          tool_output=out_text, is_error=is_err, timestamp=ts))
        elif role == "exit":
            exit_msg = msg  # last one wins; a trajectory only writes one
    events.append(_swemini_run_end(exit_msg, info, messages))
    return {"events": events, "session_meta": meta, "warnings": warnings}


# --------------------------------------------------------- claude/-pro ---

def _content_text(content):
    """Anthropic tool_result content can be a bare string or a list of blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(parts)
    return str(content)


# Claude Code records the ending in exactly one place: the final `result`
# record. `terminal_reason` is the harness's own verdict ("completed",
# "api_error", "max_turns", "blocking_limit"), `result` carries the provider's
# error text when there was one, and `stop_reason` says what the last model
# call did.
#
# The case that needs saying out loud is the turn budget. A run cut at
# EVAL_MAX_TURNS ends exactly like a run that finished: last tool call answered,
# nothing after it. The difference lives only in this record — subtype
# "error_max_turns" / is_error true / errors ["Reached maximum number of turns
# (120)"], versus subtype "success" / is_error false for a model that simply
# stopped asking for tools. Read as a timeline the two are identical, and one of
# them left a half-finished edit on disk.
#
# A killed process (walltime, SIGKILL) writes no result record at all: the
# stream just stops. That absence is itself the signal — reason `cut_off`.
def _claude_run_end(res):
    """Terminal timeline event: why this claude run stopped."""
    reason, used, limit, what = _claude_end_reason(res)
    res = res or {}
    parts = [_END_LABEL.get(reason, reason)]
    detail = ""
    if reason == "max_turns":
        errs = [e for e in (res.get("errors") or []) if isinstance(e, str)]
        detail = errs[0] if errs else ""
        turns = res.get("num_turns")
        if turns:
            parts.append(f"stopped after {turns} turns")
        if res.get("stop_reason") == "tool_use":
            parts.append("the last tool result was never acted on — whatever the "
                         "model was mid-way through is how it left the tree")
    elif reason == "cut_off":
        parts.append("the stream ends with no result record — the process was "
                     "killed (walltime or a signal) before one was written")
    elif used and limit:
        parts.append(_token_phrase(used, limit, what))
    elif reason == "output_limit":
        parts.append("the reply stopped at the model's output token cap")
    if not detail and res.get("is_error") and isinstance(res.get("result"), str):
        detail = res["result"]
    status = res.get("api_error_status")
    if status and str(status) not in detail:
        detail = f"HTTP {status}\n{detail}" if detail else f"HTTP {status}"

    text, trunc = _cap(" · ".join(parts))
    detail_text, _ = _cap(detail) if detail else (None, False)
    return _new_event(
        "run_end", role="system", text=text, text_truncated=trunc,
        tool_output=detail_text or None,
        is_error=_END_SEVERITY.get(reason, "error") == "error",
        end_reason=reason,
        end_severity=_END_SEVERITY.get(reason, "error"),
        end_stop_reason=res.get("terminal_reason") or res.get("subtype"),
        end_tokens_used=used, end_tokens_limit=limit,
    )


def _parse_claude(path):
    events = []
    warnings = []
    meta = {}
    pending_calls = {}
    result_rec = None   # absent = the process was killed before it could finish

    for d in _iter_jsonl(path):
        t = d.get("type")
        if t == "system" and d.get("subtype") == "init":
            meta.update({"session_id": d.get("session_id"), "cwd": d.get("cwd"),
                         "model": d.get("model"), "tools": d.get("tools"),
                         "claude_code_version": d.get("claude_code_version")})
        elif t == "assistant":
            msg = d.get("message", {}) or {}
            meta.setdefault("model", msg.get("model"))
            for block in msg.get("content", []) or []:
                bt = block.get("type")
                if bt == "text" and block.get("text"):
                    text, trunc = _cap(block.get("text"))
                    events.append(_new_event("assistant_text", role="assistant", text=text, text_truncated=trunc))
                elif bt == "thinking":
                    text, trunc = _cap(block.get("thinking"))
                    if text:
                        events.append(_new_event("thinking", role="assistant", text=text, text_truncated=trunc))
                elif bt == "tool_use":
                    ev = _new_event("tool_call", role="assistant", tool_name=block.get("name"),
                                     tool_call_id=block.get("id"), tool_input=block.get("input"))
                    events.append(ev)
                    if block.get("id"):
                        pending_calls[block["id"]] = ev
        elif t == "user":
            msg = d.get("message", {}) or {}
            content = msg.get("content")
            if isinstance(content, str):
                text, trunc = _cap(content)
                events.append(_new_event("user", role="user", text=text, text_truncated=trunc))
                continue
            for block in content or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    call_id = block.get("tool_use_id")
                    ev = pending_calls.get(call_id)
                    out_text, trunc = _cap(_content_text(block.get("content")))
                    if ev is not None:
                        ev["tool_output"] = out_text
                        ev["tool_output_truncated"] = trunc
                        ev["is_error"] = bool(block.get("is_error"))
                    else:
                        events.append(_new_event("log", text=f"orphan tool result for {call_id}",
                                                  tool_output=out_text, is_error=bool(block.get("is_error"))))
                elif block.get("type") == "text" and block.get("text"):
                    text, trunc = _cap(block.get("text"))
                    events.append(_new_event("user", role="user", text=text, text_truncated=trunc))
        elif t == "result":
            result_rec = d      # last one wins; a run only writes one
            meta["result"] = {
                "subtype": d.get("subtype"), "is_error": d.get("is_error"),
                "duration_ms": d.get("duration_ms"), "num_turns": d.get("num_turns"),
                "result_text": d.get("result"),
            }

    events.append(_claude_run_end(result_rec))
    return {"events": events, "session_meta": meta, "warnings": warnings}


# ---------------------------------------------------------------- kimi ---

# Why a kimi run stopped isn't in agent.stream.jsonl at all: a clean turn just
# ends after the last assistant message (closed out by a `session.resume_hint`
# meta line), while a harness-side stop — the per-turn step cap, a dropped
# connection, a provider error — is never written into the trace file at all.
# It only ever surfaces as the last line of the sibling agent.stderr.log
# (solve.sh redirects there: `kimi -p ... > agent.stream.jsonl 2>
# agent.stderr.log`). Without this the timeline just stops mid-conversation
# and looks like the model gave up — same problem the pi parser already
# solves for context overflow (_pi_run_end above).
def _kimi_run_end(round_dir, last_role, last_has_toolcalls, last_meta_type,
                  final_has_toolcalls=False, backgrounded=False):
    """Terminal timeline event: why this kimi run stopped."""
    wire = fc.read_kimi_wire(round_dir)
    reason, msg, used, limit, what = fc.kimi_end_reason(
        round_dir, last_role, last_has_toolcalls, last_meta_type,
        final_has_toolcalls, backgrounded, wire[0] if wire else None)

    parts = [_END_LABEL.get(reason, reason)]
    if reason == "max_steps_exceeded":
        m = fc.KIMI_MAX_STEPS_RE.search(msg)
        if m:
            parts.append(f"the turn hit its {m.group(1)}-step cap before it could continue")
    elif used and limit:
        parts.append(_token_phrase(used, limit, what))
    elif reason == "cut_off":
        parts.append("the run stops after the last tool call — turn limit, "
                      "walltime, or the harness stopped it")
    elif reason == "empty_response":
        parts.append("the model produced no text or tool call to continue on")
    elif reason == "connection_error":
        parts.append("the connection to the provider dropped mid-run")
    elif reason == "output_limit":
        parts.append("the last model call stopped at the server output cap")
    elif reason == "hung_after_final":
        parts.append("a background task kept kimi alive after its final answer")

    text, trunc = _cap(" · ".join(parts))
    detail_text, _ = _cap(msg) if msg else (None, False)
    return _new_event(
        "run_end", role="system", text=text, text_truncated=trunc,
        tool_output=detail_text or None,
        is_error=_END_SEVERITY.get(reason, "error") == "error",
        end_reason=reason,
        end_severity=_END_SEVERITY.get(reason, "error"),
        end_tokens_used=used, end_tokens_limit=limit,
    )


def _parse_kimi(path):
    events = []
    warnings = []
    meta = {}
    pending_calls = {}
    last_role = None
    last_has_toolcalls = final_has_toolcalls = backgrounded = False
    last_meta_type = None

    for d in _iter_jsonl(path):
        role = d.get("role")
        last_role = role
        last_has_toolcalls = False
        last_meta_type = None
        if role in ("system", "user"):
            text, trunc = _cap(d.get("content"))
            events.append(_new_event(role, role=role, text=text, text_truncated=trunc))
        elif role == "assistant":
            content = d.get("content")
            if content and content.strip():
                text, trunc = _cap(content)
                events.append(_new_event("assistant_text", role="assistant", text=text, text_truncated=trunc))
            calls = d.get("tool_calls") or []
            last_has_toolcalls = final_has_toolcalls = bool(calls)
            for call in calls:
                fn = call.get("function", {}) or {}
                args = fn.get("arguments")
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except Exception:
                    pass
                ev = _new_event("tool_call", role="assistant", tool_name=fn.get("name"),
                                 tool_call_id=call.get("id"), tool_input=args)
                events.append(ev)
                if call.get("id"):
                    pending_calls[call["id"]] = ev
        elif role == "tool":
            call_id = d.get("tool_call_id")
            ev = pending_calls.get(call_id)
            backgrounded = backgrounded or fc.KIMI_BACKGROUND_TASK in json.dumps(d.get("content"))
            out_text, trunc = _cap(d.get("content"))
            if ev is not None:
                ev["tool_output"] = out_text
                ev["tool_output_truncated"] = trunc
            else:
                events.append(_new_event("log", text=f"orphan tool result for {call_id}", tool_output=out_text))
        elif role == "meta":
            kind = d.get("type", "meta")
            last_meta_type = kind
            bits = {k: v for k, v in d.items() if k not in ("role", "type")}
            text = f"[{kind}] " + json.dumps(bits, default=str)
            text, trunc = _cap(text)
            events.append(_new_event("log", text=text, text_truncated=trunc))

    ev = _kimi_run_end(os.path.dirname(path), last_role, last_has_toolcalls, last_meta_type,
                       final_has_toolcalls, backgrounded)
    if ev is not None:
        events.append(ev)
    return {"events": events, "session_meta": meta, "warnings": warnings}
