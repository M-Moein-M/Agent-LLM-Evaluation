"""mock_backend.py PORT LOG   (binds 127.0.0.1 only)
OpenAI-compatible mock of sglang. Records every POST body to LOG (jsonl: path, body).
Non-stream reply: reasoning_content + one tool call. Stream reply: SSE chunks like sglang's,
chosen by a tag "STREAMMODE:<mode>" anywhere in the request text:
  inorder : reasoning, call0 (bash) complete, '\\n' content chunk, call1 (read), finish, usage chunk
  late    : reasoning, call0 args split, '\\n' content chunk, LATE '}' for call0, call1, finish, usage chunk
  text    : reasoning chunks, then text chunks, finish, usage chunk
  (default inorder)
Usage always has prompt_tokens=1000 with prompt_tokens_details.cached_tokens=700.
"""
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
PORT, LOG = int(sys.argv[1]), sys.argv[2]
USAGE = {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050,
         "prompt_tokens_details": {"cached_tokens": 700}}

def delta_chunk(delta, finish=None, usage=None, empty_choices=False):
    c = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0, "model": "mock",
         "choices": [] if empty_choices else [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage: c["usage"] = usage
    return c

def tc(i, name=None, args="", id=None):
    f = {"arguments": args}
    d = {"index": i, "function": f}
    if name is not None:
        f["name"] = name; d["id"] = id or f"call_mock{i}"; d["type"] = "function"
    return {"tool_calls": [d]}

def stream_chunks(mode, include_usage):
    ch = [delta_chunk({"role": "assistant", "content": None, "reasoning_content": x}) for x in ("think-A ", "think-B ", "think-C ")]
    if mode == "text":
        ch += [delta_chunk({"content": x}) for x in ("text-E ", "text-F ", "text-G ")]
        fin = "stop"
    else:
        ch += [delta_chunk(tc(0, "bash")), delta_chunk(tc(0, args='{"command":')), delta_chunk(tc(0, args=' "ls -la /tmp"'))]
        if mode == "late":
            ch += [delta_chunk({"content": "\n"}), delta_chunk(tc(0, args="}"))]
        else:
            ch += [delta_chunk(tc(0, args="}")), delta_chunk({"content": "\n"})]
        ch += [delta_chunk(tc(1, "read")), delta_chunk(tc(1, args='{"path": "/etc/hosts"}'))]
        fin = "tool_calls"
    ch.append(delta_chunk({}, finish=fin))
    if include_usage:
        ch.append(delta_chunk({}, usage=USAGE, empty_choices=True))
    return ch

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._json({"object": "list", "data": [{"id": "mock", "object": "model"}]})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        with open(LOG, "a") as f:
            f.write(json.dumps({"path": self.path, "body": body, "t": time.time()}) + "\n")
        if self.path.rstrip("/").endswith("/responses"):
            return self._json({"error": {"message": "mock: /responses not implemented (like sglang)"}}, 404)
        if body.get("stream"):
            text = json.dumps(body.get("messages", []))
            mode = "inorder"
            for m in ("late", "inorder", "text"):
                if f"STREAMMODE:{m}" in text: mode = m
            inc = bool((body.get("stream_options") or {}).get("include_usage"))
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close"); self.end_headers()
            for c in stream_chunks(mode, inc):
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode()); self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            self.close_connection = True
            return
        self._json({"id": "chatcmpl-mock", "object": "chat.completion", "created": 0, "model": body.get("model", "mock"),
                    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                        "role": "assistant", "content": None, "reasoning_content": "mock reasoning",
                        "tool_calls": [{"id": "call_mock1", "type": "function",
                                        "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}]}}],
                    "usage": USAGE})
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
