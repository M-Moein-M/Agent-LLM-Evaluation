"""proxy_probe.py PROXY_URL RUNDIR LABEL
Sends cases to the proxy on both routes (stream and non-stream), then reads RUNDIR/backend.jsonl
(what the mock backend received) and RUNDIR/hooks.jsonl (optional: what a logging callback saw).
Writes RUNDIR/probe_<LABEL>.json and prints a table."""
import json, os, sys, time, urllib.request, urllib.error
proxy, run, label = sys.argv[1:4]

# Model id the proxy was started with (start_stack.sh uses the same default), and the API key if a master key is set.
M = os.environ.get("PROBE_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
API_KEY = os.environ.get("PROXY_API_KEY", "dummy")
REASON = "I should check the weather first. " * 300
ARGS = {"city": "Paris"}
OA_TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "weather",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
AN_TOOLS = [{"name": "get_weather", "description": "weather",
             "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
TB = [{"type": "thinking", "thinking": REASON, "signature": "sig"}]


def oa(tag, extra=None, tool=True, stream=False, mode=None):
    a = {"role": "assistant", "content": None if tool else "It is cloudy.", **(extra or {})}
    msgs = [{"role": "user", "content": f"probe {tag} " + (f"STREAMMODE:{mode}" if mode else "")}, a]
    if tool:
        a["tool_calls"] = [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps(ARGS)}}]
        msgs.append({"role": "tool", "tool_call_id": "call_1", "content": "cloudy, 18C"})
    else:
        msgs.append({"role": "user", "content": "and Tokyo?"})
    b = {"model": M, "messages": msgs, "tools": OA_TOOLS, "stream": stream}
    if stream:
        b["stream_options"] = {"include_usage": True}
    return "/v1/chat/completions", b


def an(tag, thinking=True, tool=True, model=M, stream=False, mode=None):
    blocks = [{"type": "thinking", "thinking": REASON, "signature": "sig"}] if thinking else []
    if tool:
        blocks.append({"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": ARGS})
        nxt = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "cloudy, 18C"}]}
    else:
        blocks.append({"type": "text", "text": "It is cloudy."})
        nxt = {"role": "user", "content": "and Tokyo?"}
    body = {"model": model, "max_tokens": 32000, "tools": AN_TOOLS, "stream": stream,
            "thinking": {"type": "enabled", "budget_tokens": 31999},
            "messages": [{"role": "user", "content": f"probe {tag} " + (f"STREAMMODE:{mode}" if mode else "")},
                         {"role": "assistant", "content": blocks}, nxt]}
    return "/v1/messages", body


CASES = [
    ("C1", "chat/compl, reasoning not sent", oa("C1")),
    ("C2", "chat/compl, reasoning_content", oa("C2", {"reasoning_content": REASON})),
    ("C3", "chat/compl, thinking_blocks+tool", oa("C3", {"thinking_blocks": TB})),
    ("C4", "chat/compl, thinking_blocks, text", oa("C4", {"thinking_blocks": TB}, tool=False)),
    ("C5", "messages, no thinking sent", an("C5", False)),
    ("C6", "messages, thinking+tool_use", an("C6")),
    ("C7", "messages, thinking, text reply", an("C7", tool=False)),
    ("C8", "messages, thinking+tool_use, -think", an("C8", model=M + "-think")),
    ("C9", "messages, thinking+tool_use, stream", an("C9", stream=True, mode="inorder")),
    ("C10", "messages stream, call/\\n/call IN ORDER", an("C10", stream=True, mode="inorder")),
    ("C11", "messages stream, \\n BEFORE last arg (P3)", an("C11", stream=True, mode="late")),
    ("C12", "messages stream, reasoning then text (P2)", an("C12", stream=True, mode="text")),
    ("C13", "chat/compl stream, late-arg order", oa("C13", {"reasoning_content": REASON}, stream=True, mode="late")),
    ("C14", "messages stream -think, late-arg order", an("C14", model=M + "-think", stream=True, mode="late")),
    ("C15", "messages stream, other model name, text", an("C15", model="claude-sonnet-4-5", stream=True, mode="text")),
]


def parse_sse(raw):
    evs = []
    for block in raw.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                s = line[5:].strip()
                if s and s != "[DONE]":
                    try:
                        evs.append(json.loads(s))
                    except Exception:
                        evs.append({"_raw": s})
    return evs


def summarize_anthropic(evs):
    blocks, order, usage = {}, [], {}
    for ev in evs:
        t = ev.get("type")
        if t is None:
            continue
        order.append(t.replace("content_block_", "cb_") + (str(ev["index"]) if "index" in ev else ""))
        if t == "message_start":
            usage["message_start"] = ev["message"].get("usage")
        elif t == "content_block_start":
            b = ev["content_block"]
            blocks[ev["index"]] = {"type": b["type"], "name": b.get("name"), "text": b.get("text") or b.get("thinking") or ""}
        elif t == "content_block_delta":
            d = ev["delta"]
            blocks.setdefault(ev["index"], {"type": "?", "name": None, "text": ""})["text"] += d.get("text") or d.get("thinking") or d.get("partial_json") or ""
        elif t == "message_delta":
            usage["message_delta"] = ev.get("usage")
            usage["stop_reason"] = ev["delta"].get("stop_reason")
    out = []
    for k, v in blocks.items():
        s = f"{k}:{v['type']}" + (f"({v['name']!r})" if v["type"] == "tool_use" else "") + f"={v['text'][:60]!r}"
        if v["type"] == "tool_use":
            try:
                json.loads(v["text"] or "null")
            except Exception:
                s += " [JSON-BAD]"
        out.append(s)
    return out, " ".join(order), usage


status, resp = {}, {}
for tag, _, (path, body) in CASES:
    req = urllib.request.Request(proxy + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}",
                                  "x-api-key": API_KEY, "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read().decode("utf-8", "replace")
            status[tag] = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        status[tag] = f"HTTP{e.code}"
    except Exception as e:
        raw = ""
        status[tag] = f"{type(e).__name__}: {str(e)[:60]}"
    resp[tag] = raw
time.sleep(2)

got = {}
for line in open(f"{run}/backend.jsonl"):
    rec = json.loads(line)
    msgs = rec["body"].get("messages", [])
    first = msgs[0]["content"] if msgs else ""
    first = first if isinstance(first, str) else json.dumps(first)
    tag = next((t for t, _, _ in CASES if f"probe {t} " in first), None)
    if tag:
        a = next((m for m in msgs if m.get("role") == "assistant"), {})
        got[tag] = {"path": rec["path"], "assistant_keys": sorted(a.keys()),
                    "reasoning_content_chars": len(a.get("reasoning_content") or ""),
                    "thinking_blocks": bool(a.get("thinking_blocks")),
                    "top_keys": sorted(k for k in rec["body"] if k != "messages"),
                    "model": rec["body"].get("model"), "stream": rec["body"].get("stream"),
                    "stream_options": rec["body"].get("stream_options"),
                    "reasoning_effort": rec["body"].get("reasoning_effort"),
                    "chat_template_kwargs": rec["body"].get("chat_template_kwargs")}
result = {"label": label, "reasoning_sent_chars": len(REASON), "cases": {}}
print(f"== {label}: reasoning sent by client: {len(REASON)} chars")
print(f"{'id':4}{'case':44}{'http':7}{'backend path':22}{'rc chars':9}{'tb':4} backend assistant keys")
ids = [c[0] for c in CASES]
for tag, lab, (path, body) in CASES:
    g = got.get(tag, {})
    print(f"{tag:4}{lab:44}{str(status[tag])[:6]:7}{g.get('path','-'):22}{g.get('reasoning_content_chars',-1):<9}{'yes' if g.get('thinking_blocks') else 'no':4} {g.get('assistant_keys')}")
    entry = {"label": lab, "http": status[tag], "backend": g}
    if body.get("stream") and status[tag] == 200:
        evs = parse_sse(resp[tag])
        if path == "/v1/messages":
            blocks, order, usage = summarize_anthropic(evs)
            entry.update(blocks=blocks, order=order, usage=usage)
        else:
            entry["n_chunks"] = len(evs)
            entry["chunk_deltas"] = [json.dumps((e.get("choices") or [{}])[0].get("delta"))[:100] if e.get("choices") else ("usage:" + json.dumps(e.get("usage"))) for e in evs]
    else:
        entry["body"] = resp[tag][:600]
    result["cases"][tag] = entry
print()
for tag, e in result["cases"].items():
    if "blocks" in e:
        print(f"{tag} {e['label']}\n      blocks: " + "\n              ".join(e["blocks"]) + f"\n      order: {e['order']}\n      stop: {e['usage'].get('stop_reason')}\n      usage start: {e['usage'].get('message_start')}\n      usage delta: {e['usage'].get('message_delta')}")
    elif "chunk_deltas" in e:
        print(f"{tag} {e['label']}: {e['n_chunks']} chunks, deltas: {e['chunk_deltas']}")
    else:
        print(f"{tag} {e['label']}: http={e['http']} body[:300]={e.get('body','')[:300]!r}")
try:
    print("\nhook log:")
    for line in open(f"{run}/hooks.jsonl"):
        print("  " + line.rstrip()[:420])
except FileNotFoundError:
    print("(no hook log for this variant)")
json.dump(result, open(f"{run}/probe_{label}.json", "w"), indent=1)
