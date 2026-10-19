#!/bin/bash
# Build the patched Kimi Code binary that host.sh bind-mounts over /usr/local/bin/kimi.
#
#   agents/apptainer/kimi/make_patched_kimi.sh [agent.sif] [out]
#   defaults: agents/apptainer/kimi/agent.sif  ->  .run/kimi-patched
#
# Takes the stock Kimi Code 0.31.0 binary out of the image and applies two same-length
# byte edits (the binary is never committed):
#   1. `maxCompactionPerTurn: Infinity` -> `0` (2 sites). A context-overflow 400 then
#      ends the run ("Compaction limit exceeded (0)") instead of compacting and retrying,
#      which keeps the rule "no compaction, server window only" identical across agents.
#   2. OpenAI client `DEFAULT_TIMEOUT = 6e5` -> `1e8` (10 minutes -> effectively none), so
#      slow long generations are not cut off and retried.
# Fails loudly if the binary is not the expected one. A different Kimi version needs the
# trace parsers re-checked first; override the checksum guard with KIMI_PATCH_ANY_SHA=1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SIF="${1:-$ROOT/agents/apptainer/kimi/agent.sif}"
OUT="${2:-$ROOT/.run/kimi-patched}"
EXPECT_SHA256="5d0b48f1994959ff620834ab3aae54a01aa5ab0e68cd1e03c857c928414b8916"   # stock 0.31.0

[ -f "$SIF" ] || { echo "ERROR: image not found: $SIF (build agents/apptainer/kimi/agent.def first)" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
TMP="$OUT.stock.$$"
trap 'rm -f "$TMP"' EXIT
apptainer exec "$SIF" cat /usr/local/bin/kimi > "$TMP"

python3 - "$TMP" "$OUT" "$EXPECT_SHA256" <<'PY'
import hashlib, os, sys
src, out, expect = sys.argv[1:4]
data = open(src, "rb").read()
sha = hashlib.sha256(data).hexdigest()
if sha != expect and os.environ.get("KIMI_PATCH_ANY_SHA") != "1":
    sys.exit("ERROR: stock kimi sha256 %s != expected %s (not Kimi Code 0.31.0). "
             "Set KIMI_PATCH_ANY_SHA=1 to patch anyway." % (sha, expect))
edits = [  # (old, new, expected occurrences); old and new have equal length
    (b"maxCompactionPerTurn: Infinity,", b"maxCompactionPerTurn: 0       ,", 2),
    (b"OpenAI.DEFAULT_TIMEOUT = 6e5;", b"OpenAI.DEFAULT_TIMEOUT = 1e8;", 1),
]
for old, new, n in edits:
    assert len(old) == len(new)
    got = data.count(old)
    if got != n:
        sys.exit("ERROR: expected %d occurrence(s) of %r, found %d" % (n, old.decode(), got))
    data = data.replace(old, new)
tmp = out + ".tmp"
with open(tmp, "wb") as f:
    f.write(data)
os.chmod(tmp, 0o755)
os.replace(tmp, out)
print("patched kimi written to %s (sha256 %s)" % (out, hashlib.sha256(data).hexdigest()))
PY
