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
Event kinds: system | user | assistant_text | thinking | tool_call | log
(keep this list in sync with FILTER_KINDS/LABEL in static/js/timeline.js)
"""
import json
import os

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
          "is_error": False, "timestamp": None}
    ev.update(kw)
    return ev


def detect_kind(round_dir):
    """Return (kind, filename) for the trace file present in a round dir, or (None, None)."""
    for fname, kind in (("agent.jsonl", "pi"),
                        ("agent.traj.json", "swe-mini"),
                        ("agent.stream.jsonl", "stream")):
        if os.path.isfile(os.path.join(round_dir, fname)):
            if kind == "stream":
                # disambiguate claude/claude-pro vs kimi by peeking the first line
                try:
                    with open(os.path.join(round_dir, fname), "r", errors="replace") as f:
                        first = f.readline()
                    d = json.loads(first) if first.strip() else {}
                    kind = "claude" if "type" in d else "kimi"
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

def _parse_pi(path):
    events = []
    warnings = []
    meta = {}
    pending_calls = {}  # toolCallId -> event dict (index into events)

    for d in _iter_jsonl(path):
        t = d.get("type")
        if t == "session":
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
                for block in msg.get("content", []) or []:
                    bt = block.get("type")
                    if bt == "text" and block.get("text"):
                        text, trunc = _cap(block.get("text"))
                        events.append(_new_event("assistant_text", role="assistant", text=text, text_truncated=trunc, timestamp=ts))
                    elif bt == "thinking":
                        text, trunc = _cap(block.get("thinking"))
                        events.append(_new_event("thinking", role="assistant", text=text, text_truncated=trunc, timestamp=ts))
                    elif bt == "toolCall":
                        ev = _new_event("tool_call", role="assistant", tool_name=block.get("name"),
                                         tool_call_id=block.get("id"), tool_input=block.get("arguments"),
                                         timestamp=ts)
                        events.append(ev)
                        if block.get("id"):
                            pending_calls[block["id"]] = ev
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
    return {"events": events, "session_meta": meta, "warnings": warnings}


# ------------------------------------------------------------ swe-mini ---

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

    for msg in doc.get("messages", []) or []:
        role = msg.get("role")
        if role == "system":
            continue  # boilerplate agent scaffolding, not useful in the timeline
        if role == "user":
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
            extra = msg.get("extra") or {}
            status = extra.get("exit_status") if isinstance(extra, dict) else None
            events.append(_new_event("log", text=f"session exit: {status}"))
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


def _parse_claude(path):
    events = []
    warnings = []
    meta = {}
    pending_calls = {}

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
            meta["result"] = {
                "subtype": d.get("subtype"), "is_error": d.get("is_error"),
                "duration_ms": d.get("duration_ms"), "num_turns": d.get("num_turns"),
                "result_text": d.get("result"),
            }
    return {"events": events, "session_meta": meta, "warnings": warnings}


# ---------------------------------------------------------------- kimi ---

def _parse_kimi(path):
    events = []
    warnings = []
    meta = {}
    pending_calls = {}

    for d in _iter_jsonl(path):
        role = d.get("role")
        if role in ("system", "user"):
            text, trunc = _cap(d.get("content"))
            events.append(_new_event(role, role=role, text=text, text_truncated=trunc))
        elif role == "assistant":
            content = d.get("content")
            if content and content.strip():
                text, trunc = _cap(content)
                events.append(_new_event("assistant_text", role="assistant", text=text, text_truncated=trunc))
            for call in d.get("tool_calls") or []:
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
            out_text, trunc = _cap(d.get("content"))
            if ev is not None:
                ev["tool_output"] = out_text
                ev["tool_output_truncated"] = trunc
            else:
                events.append(_new_event("log", text=f"orphan tool result for {call_id}", tool_output=out_text))
        elif role == "meta":
            kind = d.get("type", "meta")
            bits = {k: v for k, v in d.items() if k not in ("role", "type")}
            text = f"[{kind}] " + json.dumps(bits, default=str)
            text, trunc = _cap(text)
            events.append(_new_event("log", text=text, text_truncated=trunc))
    return {"events": events, "session_meta": meta, "warnings": warnings}
