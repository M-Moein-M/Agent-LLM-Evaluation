#!/usr/bin/env python3
"""Builds the static frontend into a self-contained, cache-busted bundle.

There is no npm in this project on purpose, so this is the whole toolchain:
concatenate the vendored libs and our modules in load order, hash the result,
emit `bundle.<hash>.js` / `style.<hash>.css`, and rewrite index.html to point at
them. Stdlib only, no network — it runs inside the Apptainer %post.

    python3 build_web.py --out /opt/trace-vis/static
    python3 build_web.py --out DIR --check     # verify an existing build

Serving the source tree directly still works (that's the dev path); the built
tree is what ships in the image, selected via TRACE_VIS_STATIC.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "static")

# Load order matters: marked and DOMPurify must exist before markdown.js runs,
# and app.js calls into markdown.js/timeline.js at definition time.
JS_ORDER = [
    "vendor/marked.min.js",
    "vendor/purify.min.js",
    "js/markdown.js",
    "js/timeline.js",
    "js/app.js",
]
CSS_FILE = "css/style.css"
VERSION = "1.0.0"


def _read(rel):
    with open(os.path.join(SRC, rel), encoding="utf-8") as f:
        return f.read()


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(out_dir):
    missing = [r for r in JS_ORDER + [CSS_FILE, "index.html"]
               if not os.path.isfile(os.path.join(SRC, r))]
    if missing:
        sys.exit(f"build_web: missing source files: {missing}")

    parts = []
    for rel in JS_ORDER:
        src = _read(rel)
        # `;` guards against a vendored file that omits its trailing semicolon
        parts.append(f"/* --- {rel} --- */\n{src}\n;\n")
    bundle_js = "".join(parts)
    css = _read(CSS_FILE)

    js_hash = _sha(bundle_js)[:10]
    css_hash = _sha(css)[:10]
    js_name = f"bundle.{js_hash}.js"
    css_name = f"style.{css_hash}.css"

    html = _read("index.html")
    # collapse the script tags into the single bundle, keeping the first one's spot
    html, n = re.subn(r'[ \t]*<script src="/(?:vendor|js)/[^"]+"></script>\n',
                      "", html)
    if n != len(JS_ORDER):
        sys.exit(f"build_web: index.html referenced {n} scripts, expected {len(JS_ORDER)}")
    html = html.replace("</body>", f'<script src="/{js_name}"></script>\n</body>')
    html, n = re.subn(r'href="/css/style\.css"', f'href="/{css_name}"', html)
    if n != 1:
        sys.exit("build_web: could not rewrite the stylesheet link in index.html")

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, js_name), "w", encoding="utf-8") as f:
        f.write(bundle_js)
    with open(os.path.join(out_dir, css_name), "w", encoding="utf-8") as f:
        f.write(css)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    info = {
        "version": VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bundle": js_name,
        "stylesheet": css_name,
        "sources": {rel: _sha(_read(rel))[:12] for rel in JS_ORDER + [CSS_FILE, "index.html"]},
        "bundle_bytes": len(bundle_js.encode("utf-8")),
    }
    with open(os.path.join(out_dir, "build-info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    check(out_dir)
    print(f"build_web: {out_dir}  {js_name} ({info['bundle_bytes'] // 1024} KB) + {css_name}")
    return info


def check(out_dir):
    """Assert the built tree is internally consistent — a bad build must not ship."""
    index = os.path.join(out_dir, "index.html")
    if not os.path.isfile(index):
        sys.exit(f"build_web: no index.html in {out_dir}")
    html = open(index, encoding="utf-8").read()
    refs = re.findall(r'(?:src|href)="(/[^"]+)"', html)
    if not refs:
        sys.exit("build_web: built index.html references nothing")
    for ref in refs:
        p = os.path.join(out_dir, ref.lstrip("/"))
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            sys.exit(f"build_web: built index.html references a missing asset: {ref}")
    js = [r for r in refs if r.endswith(".js")]
    if len(js) != 1:
        sys.exit(f"build_web: expected exactly one bundle, got {js}")
    body = open(os.path.join(out_dir, js[0].lstrip("/")), encoding="utf-8").read()
    for marker in ("DOMPurify", "window.TV.md", "window.TV.renderTimeline", "hashchange"):
        if marker not in body:
            sys.exit(f"build_web: bundle is missing {marker!r} — load order or a source file is wrong")
    print(f"build_web: check ok ({len(refs)} assets)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the trace-vis frontend bundle.")
    ap.add_argument("--out", required=True, help="output directory (replaced if it exists)")
    ap.add_argument("--check", action="store_true", help="verify an existing build instead")
    args = ap.parse_args(argv)
    if args.check:
        check(args.out)
    else:
        build(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
