#!/bin/bash
# Run a problem's starter code through its checker without any agent.
#
#   scripts/check_starter.sh <problem> "<run command>" [build command, default: make]
#   e.g. scripts/check_starter.sh _template "./prog -c"
#
# Copies problems/<problem> to a temp dir, builds and runs it with the host toolchain, writes the
# same evidence measure.sh writes (.eval/facts.json, run.log) into a fake trace, and runs
# checks/check_pass.py on it. For GPU problems run it inside an agent image:
#   apptainer exec --nv agents/apptainer/pi/agent.sif scripts/check_starter.sh tiled-gemm "./gemm -c"
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="${1:?usage: check_starter.sh <problem> \"<run command>\" [build command]}"
RUN="${2:?run command required}"
BUILD="${3:-make}"
[ -d "$ROOT/problems/$P" ] || { echo "no problems/$P" >&2; exit 2; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
W="$T/work"; E="$T/traces/$P/starter/starter/check/round-1/.eval"
mkdir -p "$E"; cp -a "$ROOT/problems/$P" "$W"; cd "$W" || exit 2

t0=$(date +%s); bash -c "$BUILD" > "$E/build.log" 2>&1; bs=$?; t1=$(date +%s)
rs=-1; if [ $bs -eq 0 ]; then bash -c "$RUN" > "$E/run.log" 2>&1; rs=$?; fi; t2=$(date +%s)
nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits > "$E/gpu_usage.csv" 2>/dev/null || true
cat > "$E/facts.json" <<JSON
{"model": "starter", "agent": "starter", "round": 1, "date": "check", "size": 0, "variant": "$P", "source": "check_starter",
 "build": {"cmd": "$BUILD", "status": $bs, "start": $t0, "end": $t1},
 "run": {"cmd": "$RUN", "status": $rs, "start": $t1, "end": $t2}}
JSON
echo "build exit=$bs run exit=$rs (logs: build.log, run.log are in the temp dir, tail of run.log below)"
tail -5 "$E/run.log" 2>/dev/null
python3 "$ROOT/checks/check_pass.py" --traces "$T/traces" --problems "$P" >/dev/null 2>&1
python3 - "$E/pass_result.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except OSError:
    sys.exit("no pass_result.json: checks/<problem>/check.py missing or crashed")
print("passed:", d["passed"])
for s in d["stages"]:
    print("  %-8s %-5s %s" % (s["name"], "ok" if s["passed"] else ("skip" if s["skipped"] else "FAIL"), s["detail"]))
PY
