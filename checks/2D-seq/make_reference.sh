#! /bin/bash
# Regenerate checks/2D-seq/reference.json from the pristine problem.
#
# Builds an untouched copy of problems/2D-seq in a scratch directory and runs it
# with the shipped poisson.par, then writes the DIGEST line out as JSON. Run
# this whenever poisson.par changes. The digest is portable across compilers and
# ISAs to ~1e-15 relative (SOR is a contraction, so reassociation does not
# amplify), well inside the 1e-9 tolerance in check.py — regenerating on the
# target node is optional, not required.
#
#   bash checks/2D-seq/make_reference.sh          # from the repo root
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBLEM="$ROOT/problems/2D-seq"
OUT="$ROOT/checks/2D-seq/reference.json"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cp -a "$PROBLEM"/. "$WORK"/
cd "$WORK"
rm -rf GCC CLANG ICX exe-* solution.json .eval

echo "building pristine 2D-seq in $WORK ..."
make >/dev/null

echo "running (this is the serial baseline — expect a few minutes) ..."
LINE=$(./exe-GCC poisson.par | grep '^DIGEST ')
echo "$LINE"

python3 - "$LINE" "$OUT" <<'PY'
import json, sys
line, out = sys.argv[1], sys.argv[2]
fields = dict(tok.split("=", 1) for tok in line.split()[1:])
ref = {k: float(v) for k, v in fields.items()}
with open(out, "w") as fh:
    json.dump(ref, fh, indent=2)
    fh.write("\n")
print("wrote", out)
PY
