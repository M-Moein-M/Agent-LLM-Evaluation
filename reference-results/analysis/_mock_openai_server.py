#!/usr/bin/env python3
"""Throwaway local HTTP stub for capturing Kimi Code CLI's outbound request.

Listens on 127.0.0.1:8765 (or --port), accepts any POST, writes the raw
request body to --log-out (last request wins — Kimi may retry), and replies
with a minimal valid OpenAI-compat chat-completion so the CLI exits cleanly
after one call. Zero third-party dependencies (stdlib http.server only) —
Phase 0 capture step, see reference-results/analysis/kimi_token_estimate.py plan.
"""

import argparse
import http.server
import json
import time


def make_handler(log_out):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            with open(log_out, "wb") as fh:
                fh.write(body)

            reply = {
                "id": "mock-capture",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "mock-capture-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
            payload = json.dumps(reply).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            pass  # keep stdout clean for the SLURM job log

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log-out", required=True)
    args = ap.parse_args()

    server = http.server.HTTPServer(("127.0.0.1", args.port), make_handler(args.log_out))
    server.serve_forever()


if __name__ == "__main__":
    main()
