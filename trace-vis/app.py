#!/usr/bin/env python3
"""trace-vis — a small local web UI for reading agent trajectories from traces/.

Stdlib only, on purpose: the image build can't break on a pip mirror, and the
server runs anywhere a python3 exists (host or container, no venv).

    python3 app.py --workspace /path/to/nhr-eval-workspace --port 8080
"""
import argparse
import json
import mimetypes
import os
import posixpath
import re
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curation                     # noqa: E402
import parsers                      # noqa: E402
from indexer import TraceIndex      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# Source tree by default (dev); the image sets TRACE_VIS_STATIC to the bundle
# produced by build_web.py during %post.
STATIC_DIR = os.environ.get("TRACE_VIS_STATIC") or os.path.join(HERE, "static")

# Side files a run view may fetch. Whitelisted so the endpoint can never be
# turned into an arbitrary-file reader.
ALLOWED_FILES = {
    "solution.json": "solution.json",
    "changed_diff.txt": "changed_diff.txt",
    "agent.stderr.log": "agent.stderr.log",
    "facts.json": os.path.join(".eval", "facts.json"),
    "pass_result.json": os.path.join(".eval", "pass_result.json"),
    "stats.json": os.path.join(".eval", "stats.json"),
    "build.log": os.path.join(".eval", "build.log"),
    "run.log": os.path.join(".eval", "run.log"),
    "host_info.txt": os.path.join(".eval", "host_info.txt"),
    "gpu_info.txt": os.path.join(".eval", "gpu_info.txt"),
    "perf.log": os.path.join(".eval", "perf.log"),
}
MAX_FILE_BYTES = 512 * 1024


class TraceCache:
    """Parsed traces keyed by (path, mtime). Small: entries can be ~MBs."""

    def __init__(self, maxsize=6):
        self.maxsize = maxsize
        self._lock = threading.Lock()
        self._data = {}
        self._order = []

    def get(self, round_dir):
        kind, fname = parsers.detect_kind(round_dir)
        if kind is None:
            return parsers.parse_trace(round_dir)
        path = os.path.join(round_dir, fname)
        try:
            key = (path, os.path.getmtime(path))
        except OSError:
            return parsers.parse_trace(round_dir)
        with self._lock:
            if key in self._data:
                self._order.remove(key)
                self._order.append(key)
                return self._data[key]
        result = parsers.parse_trace(round_dir)
        with self._lock:
            self._data[key] = result
            self._order.append(key)
            while len(self._order) > self.maxsize:
                self._data.pop(self._order.pop(0), None)
        return result


class App:
    def __init__(self, workspace, scan_fs=False):
        self.index = TraceIndex(workspace).build(scan_fs=scan_fs)
        self.cache = TraceCache()
        self.workspace = os.path.abspath(workspace)

    # -- helpers ---------------------------------------------------------

    def _run_dir(self, q):
        run_id = (q.get("run") or [None])[0]
        if not run_id:
            return None, None, (400, "missing ?run=")
        run = self.index.get(run_id)
        if run is None:
            return None, None, (404, f"unknown run {run_id}")
        path = self.index.round_path(run_id)
        if path is None:
            return None, None, (404, f"round dir missing on disk for {run_id}")
        return run, path, None

    # -- endpoints -------------------------------------------------------

    @staticmethod
    def _scope(q):
        """?scope=curated (default) | all — see curation.py."""
        return curation.normalize_scope((q.get("scope") or [None])[0])

    def api_summary(self, q):
        self.index.maybe_refresh()
        return self.index.summary(self._scope(q))

    def api_problem(self, q):
        name = (q.get("name") or [None])[0]
        view = self.index.problem_view(name, self._scope(q)) if name else None
        return view or (404, f"unknown problem {name}")

    def api_agent(self, q):
        name = (q.get("name") or [None])[0]
        view = self.index.agent_view(name, self._scope(q)) if name else None
        return view or (404, f"unknown agent {name}")

    def api_run(self, q):
        run, path, err = self._run_dir(q)
        if err:
            return err
        kind, fname = parsers.detect_kind(path)
        present = []
        for label, rel in ALLOWED_FILES.items():
            fp = os.path.join(path, rel)
            if os.path.isfile(fp):
                try:
                    present.append({"name": label, "bytes": os.path.getsize(fp)})
                except OSError:
                    pass
        solution = None
        sp = os.path.join(path, "solution.json")
        if os.path.isfile(sp):
            try:
                with open(sp, errors="replace") as f:
                    solution = json.load(f)
            except Exception as e:
                solution = {"_error": f"unreadable: {e!r}"}
        return {"run": run, "schema": kind, "trace_file": fname,
                "solution": solution, "files": sorted(present, key=lambda d: d["name"]),
                "round_rel": run["round_rel"],
                # a run is always openable by link; the view says so when it is
                # outside the curated scope
                "curated": run["run_id"] in self.index.curated}

    def api_trace(self, q):
        run, path, err = self._run_dir(q)
        if err:
            return err
        parsed = self.cache.get(path)
        offset = int((q.get("offset") or [0])[0] or 0)
        limit = int((q.get("limit") or [parsers.MAX_EVENTS])[0] or parsers.MAX_EVENTS)
        kinds = (q.get("kinds") or [None])[0]
        kinds = set(kinds.split(",")) if kinds else None
        page = parsers.slice_events(parsed, offset, limit, kinds)
        page["run_id"] = run["run_id"]
        return page

    def api_file(self, q):
        run, path, err = self._run_dir(q)
        if err:
            return err
        name = (q.get("name") or [None])[0]
        rel = ALLOWED_FILES.get(name)
        if rel is None:
            return 400, f"file {name!r} not in the allowed set"
        fp = os.path.join(path, rel)
        if not os.path.isfile(fp):
            return 404, f"{name} not present in this round"
        size = os.path.getsize(fp)
        with open(fp, "r", errors="replace") as f:
            content = f.read(MAX_FILE_BYTES)
        return {"name": name, "bytes": size, "content": content,
                "truncated": size > MAX_FILE_BYTES}

    def api_reindex(self, q):
        self.index.build(scan_fs=self.index.scanned_fs)
        return {"ok": True, **self.index.summary()["totals"]}


ROUTES = {
    "/api/summary": "api_summary",
    "/api/problem": "api_problem",
    "/api/agent": "api_agent",
    "/api/run": "api_run",
    "/api/trace": "api_trace",
    "/api/file": "api_file",
    "/api/reindex": "api_reindex",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "trace-vis"
    app = None

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))

    def do_POST(self):
        self.do_GET()

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path in ROUTES:
                q = parse_qs(parsed.query)
                result = getattr(self.app, ROUTES[path])(q)
                if isinstance(result, tuple):
                    return self._json({"error": result[1]}, status=result[0])
                return self._json(result)
            return self._static(path)
        except BrokenPipeError:
            pass
        except Exception:
            traceback.print_exc()
            try:
                self._json({"error": "internal error"}, status=500)
            except Exception:
                pass

    def _json(self, obj, status=200):
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path):
        if path == "/" or path == "":
            path = "/index.html"
        # normalize and confine to STATIC_DIR
        rel = posixpath.normpath(path).lstrip("/")
        fp = os.path.abspath(os.path.join(STATIC_DIR, rel))
        if not fp.startswith(os.path.abspath(STATIC_DIR) + os.sep) and fp != os.path.abspath(STATIC_DIR):
            return self._json({"error": "forbidden"}, status=403)
        if not os.path.isfile(fp):
            # unknown non-API path -> SPA shell, so deep links/refresh work
            fp = os.path.join(STATIC_DIR, "index.html")
            if not os.path.isfile(fp):
                return self._json({"error": "not found"}, status=404)
        ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        with open(fp, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serve the agent trace viewer.")
    ap.add_argument("--workspace", default=os.environ.get("TRACE_VIS_WORKSPACE", "/workspace"),
                    help="repo root containing traces/ and results/ (default: /workspace)")
    ap.add_argument("--host", default=os.environ.get("TRACE_VIS_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("TRACE_VIS_PORT", "8080")))
    ap.add_argument("--scan-fs", action="store_true",
                    help="also walk traces/ for rounds with no eval result (slow on cluster FS)")
    ap.add_argument("--static", default=None,
                    help="directory to serve the frontend from (default: $TRACE_VIS_STATIC or ./static)")
    args = ap.parse_args(argv)

    if args.static:
        global STATIC_DIR
        STATIC_DIR = os.path.abspath(args.static)
    if not os.path.isfile(os.path.join(STATIC_DIR, "index.html")):
        sys.exit(f"error: no index.html under {STATIC_DIR} — pass --static or run build_web.py")

    ws = os.path.abspath(args.workspace)
    if not os.path.isdir(os.path.join(ws, "traces")):
        sys.exit(f"error: {ws}/traces not found — pass --workspace or bind it into the container")

    print(f"trace-vis: indexing {ws} ...", file=sys.stderr)
    app = App(ws, scan_fs=args.scan_fs)
    t = app.index.summary("all")["totals"]
    print(f"trace-vis: {t['n_runs']} runs, {t['n_campaigns']} campaigns, "
          f"{t['n_problems']} problems, {t['n_agents']} agents, {t['n_models']} models",
          file=sys.stderr)
    print(f"trace-vis: {len(app.index.curated)} of them are in the default "
          f"'important runs' scope (latest campaign per pairing; see curation.py)",
          file=sys.stderr)
    for w in app.index.warnings[:5]:
        print(f"trace-vis: warning: {w}", file=sys.stderr)

    Handler.app = app
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"trace-vis: serving on http://{shown}:{args.port}  (ctrl-c to stop)", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\ntrace-vis: bye", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
