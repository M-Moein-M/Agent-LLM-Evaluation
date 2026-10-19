#!/usr/bin/env python3
"""Extract the system message(s) + tools schema from a captured Kimi Code
CLI request (see reference-results/analysis/_mock_openai_server.py) into a small JSON file that
reference-results/analysis/kimi_token_estimate.py tokenizes once as the constant CLI-injected
prefix. Exits non-zero if the capture is missing or has no system message,
so the caller (scripts/_run_kimi_token_estimate.slurm) can fall back to
running without it.
"""

import json
import sys


def main():
    if len(sys.argv) != 3:
        print("usage: _extract_kimi_prefix.py <captured_request.json> <out.json>", file=sys.stderr)
        return 2

    capture_path, out_path = sys.argv[1], sys.argv[2]

    try:
        with open(capture_path, "r", encoding="utf-8") as fh:
            request = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"failed to read/parse {capture_path}: {exc}", file=sys.stderr)
        return 1

    messages = request.get("messages", [])
    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    system_text = "\n".join(p for p in system_parts if isinstance(p, str))

    if not system_text:
        print(f"no system message found in {capture_path}", file=sys.stderr)
        return 1

    tools = request.get("tools", [])

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"system": system_text, "tools": tools}, fh, indent=2)

    print(f"wrote {out_path}: system={len(system_text)} chars, tools={len(tools)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
