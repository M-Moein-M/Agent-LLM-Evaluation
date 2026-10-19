#!/usr/bin/env python3
"""L0 endpoint probes: protocol-level checks with no agent involved.

    python3 tests/endpoint/probe.py --endpoint NAME --model KEY [--json OUT]

Checks the things the four agents depend on: chat completions (plain and streamed),
tool calls (plain, streamed, parallel), reasoning output, and the Anthropic
/v1/messages surface (plain, streamed tool use, thinking blocks). Prints one line per
probe (PASS / WARN / FAIL / SKIP) and the tokens spent. Exit code 1 if any probe FAILs.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "lib"))
import resolve_endpoint as rl  # noqa: E402

TOOL_OAI = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
TOOL_ANT = {"name": "get_weather", "description": "Get the current weather for a city.",
            "input_schema": TOOL_OAI["function"]["parameters"]}
MARKUP = ("<tool_call>", "<function=", "<|tool_call", "[TOOL_CALLS]", '"name": "get_weather"')


class Http:
    def __init__(self, ep, timeout):
        self.base, self.key, self.timeout = ep["base_url"], rl.require_key(ep), timeout
        self.root = self.base[:-3] if self.base.endswith("/v1") else self.base

    def post(self, url, body, extra=None, stream=False):
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self.key,
                   "x-api-key": self.key, "anthropic-version": "2023-06-01"}
        headers.update(extra or {})
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            r = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raise RuntimeError("HTTP %d %s" % (e.code, e.read(300).decode("utf-8", "replace").strip()))
        except Exception as e:
            raise RuntimeError("%s" % e)
        if not stream:
            return json.loads(r.read().decode())
        return self._sse(r)

    @staticmethod
    def _sse(resp):
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload == "[DONE]":
                    yield "[DONE]"
                    continue
                try:
                    yield json.loads(payload)
                except ValueError:
                    pass


class Report:
    def __init__(self):
        self.rows, self.tok_in, self.tok_out = [], 0, 0

    def add(self, pid, status, detail):
        self.rows.append({"probe": pid, "status": status, "detail": detail})
        print("  %-5s %-4s %s" % (status, pid, detail))
        sys.stdout.flush()

    def tokens(self, tin, tout):
        self.tok_in += int(tin or 0)
        self.tok_out += int(tout or 0)


def chat_body(model, messages, **kw):
    b = {"model": model, "messages": messages}
    b.update(kw)
    return b


def reasoning_of(msg):
    return (msg.get("reasoning_content") or msg.get("reasoning") or "") if isinstance(msg, dict) else ""


def run(ep, model, declared_reasoning, timeout):
    http, rep = Http(ep, timeout), Report()
    url = ep["base_url"] + "/chat/completions"
    saw_reasoning = False

    def guarded(pid, fn):
        try:
            fn()
        except Exception as e:  # network / HTTP / parse failure in the probe itself
            rep.add(pid, "FAIL", str(e)[:220])

    # P1 plain completion
    def p1():
        nonlocal saw_reasoning
        r = http.post(url, chat_body(model, [{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=256))
        m = r["choices"][0]["message"]
        u = r.get("usage") or {}
        rep.tokens(u.get("prompt_tokens"), u.get("completion_tokens"))
        saw_reasoning = saw_reasoning or bool(reasoning_of(m))
        if not (m.get("content") or reasoning_of(m)):
            rep.add("P1", "FAIL", "empty reply (no content, no reasoning)")
        elif not u.get("prompt_tokens"):
            rep.add("P1", "WARN", "reply ok but usage missing/zero (token accounting will be empty)")
        else:
            rep.add("P1", "PASS", "chat completion ok, usage in=%s out=%s" % (u.get("prompt_tokens"), u.get("completion_tokens")))
    guarded("P1", p1)
    if rep.rows[0]["status"] == "FAIL" and not rep.rows[0]["detail"].startswith("HTTP"):
        for pid in ("P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"):
            rep.add(pid, "SKIP", "endpoint unreachable (see P1)")
        return rep

    # P2 streamed completion with usage
    def p2():
        nonlocal saw_reasoning
        chunks, usage, done, text = 0, None, False, 0
        for ev in http.post(url, chat_body(model, [{"role": "user", "content": "Count from 1 to 5."}], max_tokens=256,
                                           stream=True, stream_options={"include_usage": True}), stream=True):
            if ev == "[DONE]":
                done = True
                continue
            chunks += 1
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                text += len(d.get("content") or "")
                if reasoning_of(d):
                    saw_reasoning = True
        if usage:
            rep.tokens(usage.get("prompt_tokens"), usage.get("completion_tokens"))
        if chunks == 0 or not (text or saw_reasoning):
            rep.add("P2", "FAIL", "stream produced no content deltas")
        elif not done:
            rep.add("P2", "WARN", "no [DONE] terminator")
        elif not usage:
            rep.add("P2", "WARN", "stream ok but no usage chunk (stream_options.include_usage ignored)")
        else:
            rep.add("P2", "PASS", "stream ok (%d chunks), usage in=%s out=%s" % (chunks, usage.get("prompt_tokens"), usage.get("completion_tokens")))
    guarded("P2", p2)

    # P3 tool call, plain
    ask = [{"role": "user", "content": "What is the weather in Paris? Call the get_weather tool."}]

    def p3():
        r = http.post(url, chat_body(model, ask, tools=[TOOL_OAI], tool_choice="auto", max_tokens=2048))
        u = r.get("usage") or {}
        rep.tokens(u.get("prompt_tokens"), u.get("completion_tokens"))
        m = r["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        if calls:
            try:
                args = json.loads(calls[0]["function"]["arguments"])
                assert "city" in args
                rep.add("P3", "PASS", "structured tool call %s(%s)" % (calls[0]["function"]["name"], json.dumps(args)))
            except (ValueError, AssertionError, KeyError):
                rep.add("P3", "FAIL", "tool_calls present but arguments are not valid JSON with `city`: %r" % calls[0]["function"].get("arguments"))
        elif any(t in (m.get("content") or "") for t in MARKUP):
            rep.add("P3", "FAIL", "tool call leaked into message content (server tool-call parser missing or wrong)")
        else:
            rep.add("P3", "WARN", "model answered without calling the tool")
    guarded("P3", p3)

    # P4 tool call, streamed (argument deltas must assemble into valid JSON)
    def p4():
        acc, leaked = {}, ""
        usage = None
        for ev in http.post(url, chat_body(model, ask, tools=[TOOL_OAI], tool_choice="auto", max_tokens=2048,
                                           stream=True, stream_options={"include_usage": True}), stream=True):
            if ev == "[DONE]":
                continue
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                leaked += d.get("content") or ""
                for tc in d.get("tool_calls") or []:
                    slot = acc.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                    f = tc.get("function") or {}
                    slot["name"] += f.get("name") or ""
                    slot["args"] += f.get("arguments") or ""
        if usage:
            rep.tokens(usage.get("prompt_tokens"), usage.get("completion_tokens"))
        if acc:
            s = acc[min(acc)]
            try:
                assert "city" in json.loads(s["args"])
                rep.add("P4", "PASS", "streamed tool call assembles: %s(%s)" % (s["name"], s["args"]))
            except (ValueError, AssertionError):
                rep.add("P4", "FAIL", "streamed tool-call arguments do not assemble into JSON: %r" % s["args"])
        elif any(t in leaked for t in MARKUP):
            rep.add("P4", "FAIL", "streamed tool call leaked into content")
        else:
            rep.add("P4", "WARN", "no tool call in the streamed reply")
    guarded("P4", p4)

    # P5 parallel tool calls
    def p5():
        q = [{"role": "user", "content": "I need the weather in Paris and in Berlin. Call get_weather once for each city, in the same turn."}]
        r = http.post(url, chat_body(model, q, tools=[TOOL_OAI], tool_choice="auto", max_tokens=2048))
        u = r.get("usage") or {}
        rep.tokens(u.get("prompt_tokens"), u.get("completion_tokens"))
        n = len(r["choices"][0]["message"].get("tool_calls") or [])
        rep.add("P5", "PASS" if n >= 2 else "WARN", "%d tool call(s) in one reply%s" % (n, "" if n >= 2 else " (agents still work; they issue calls one by one)"))
    guarded("P5", p5)

    # P6 reasoning
    if declared_reasoning:
        rep.add("P6", "PASS" if saw_reasoning else "WARN",
                "reasoning field returned" if saw_reasoning else
                "model is declared `reasoning: true` but no reasoning_content/reasoning came back (thinking disabled in the chat template, or the server drops it)")
    else:
        rep.add("P6", "SKIP", "model not declared as reasoning")

    # Anthropic surface (claude agent)
    if "anthropic" not in ep["protocols"]:
        for pid in ("P7", "P8", "P9"):
            rep.add(pid, "SKIP", "endpoint does not list the anthropic protocol")
        return rep
    murl = http.root + "/v1/messages"

    def p7():
        r = http.post(murl, {"model": model, "max_tokens": 128, "messages": [{"role": "user", "content": "Reply with the single word: ok"}]})
        u = r.get("usage") or {}
        rep.tokens((u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0), u.get("output_tokens"))
        if not r.get("content"):
            rep.add("P7", "FAIL", "/v1/messages returned no content blocks")
        elif not (u.get("input_tokens") or u.get("output_tokens")):
            rep.add("P7", "WARN", "/v1/messages ok but usage is zero (claude token totals come from the final result event)")
        else:
            rep.add("P7", "PASS", "/v1/messages ok, usage in=%s out=%s" % (u.get("input_tokens"), u.get("output_tokens")))
    guarded("P7", p7)

    def p8():
        blocks, jsons, usage_out, usage_in = {}, {}, 0, 0
        body = {"model": model, "max_tokens": 2048, "stream": True, "tools": [TOOL_ANT],
                "messages": [{"role": "user", "content": "What is the weather in Paris? Call the get_weather tool."}]}
        for ev in http.post(murl, body, stream=True):
            if ev == "[DONE]":
                continue
            t = ev.get("type")
            if t == "message_start":
                usage_in = (ev["message"].get("usage") or {}).get("input_tokens") or 0
            elif t == "content_block_start":
                blocks[ev["index"]] = ev["content_block"].get("type")
            elif t == "content_block_delta":
                d = ev["delta"]
                if d.get("type") == "input_json_delta":
                    jsons[ev["index"]] = jsons.get(ev["index"], "") + (d.get("partial_json") or "")
            elif t == "message_delta":
                usage_out = (ev.get("usage") or {}).get("output_tokens") or usage_out
        rep.tokens(usage_in, usage_out)
        tool_idx = [i for i, b in blocks.items() if b == "tool_use"]
        if not tool_idx:
            rep.add("P8", "WARN", "no tool_use block in the streamed /v1/messages reply (blocks: %s)" % sorted(set(blocks.values())))
            return
        try:
            assert "city" in json.loads(jsons.get(tool_idx[0], ""))
            rep.add("P8", "PASS", "streamed tool_use assembles from input_json_delta: %s" % jsons[tool_idx[0]])
        except (ValueError, AssertionError):
            rep.add("P8", "FAIL", "tool_use input_json_delta does not assemble into JSON: %r" % jsons.get(tool_idx[0]))
    guarded("P8", p8)

    def p9():
        if not declared_reasoning:
            rep.add("P9", "SKIP", "model not declared as reasoning")
            return
        r = http.post(murl, {"model": model, "max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 1024},
                             "messages": [{"role": "user", "content": "What is 17 * 23? Think it through, then answer."}]})
        u = r.get("usage") or {}
        rep.tokens(u.get("input_tokens"), u.get("output_tokens"))
        has = any(b.get("type") == "thinking" and (b.get("thinking") or "").strip() for b in r.get("content") or [])
        rep.add("P9", "PASS" if has else "WARN",
                "thinking block returned on /v1/messages" if has else
                "no thinking block on /v1/messages (claude traces will show no thinking; try the endpoint's `-think` alias via agent_ids)")
    guarded("P9", p9)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=os.environ.get("EVAL_ENDPOINT"))
    ap.add_argument("--model", default=os.environ.get("EVAL_MODELS"), help="model key or id")
    ap.add_argument("--agent", default=None, help="resolve the per-agent wire id (e.g. claude)")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--json", help="write the report here")
    a = ap.parse_args()
    try:
        eps, _ = rl.load_config()
        ep, chosen = rl.select(eps, a.endpoint, [a.model] if a.model else None, a.agent)
    except rl.ConfigError as e:
        sys.exit("ERROR: %s" % e)
    c = chosen[0]
    print("L0 probes: %s  model=%s (key %s)  reasoning=%s" % (ep["base_url"], c["id"], c["key"], c["reasoning"]))
    rep = run(ep, c["id"], c["reasoning"], a.timeout)
    counts = {s: sum(1 for r in rep.rows if r["status"] == s) for s in ("PASS", "WARN", "FAIL", "SKIP")}
    print("L0 summary: %s   tokens in=%d out=%d" % (" ".join("%s=%d" % kv for kv in counts.items()), rep.tok_in, rep.tok_out))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"probes": rep.rows, "tokens_in": rep.tok_in, "tokens_out": rep.tok_out}, f, indent=2)
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
