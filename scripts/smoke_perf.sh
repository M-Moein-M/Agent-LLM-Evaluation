#! /bin/bash -l
# Smoke test for the tiled-gemm PERF-GRADING path (bench.cu -> nsys -> cuBLAS
# -> % of cuBLAS in check.py). Runs ONLY the measure phase, using the pristine
# starter kernel (src/gemm.cu) as a stand-in "agent solution" — no LLM, fully
# deterministic. Mirrors agents/apptainer/pi/host.sh:measure_job exactly.
#
# Run on an A40 node, e.g.:
#   salloc --gres=gpu:a40:1 --time=00:20:00 --partition=a40
#   bash scripts/smoke_perf.sh
#
# Expected: a naive tiled kernel lands well under the 15% cuBLAS floor, so the
# perf stage FAILS — that is the correct signal (the gate rejects unoptimized
# kernels). Success of the SMOKE TEST = the pipeline produced real numbers:
# .eval/perf.log shows BENCH lines + a "% of cuBLAS" verdict, and (if nsys is in
# the image) .eval/kern_agent.csv / kern_cublas.csv exist.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIF="${AGENT_SIF:-$ROOT/agents/apptainer/pi/agent.sif}"
MEASURE="$ROOT/agents/docker/pi/measure.sh"
BENCH="$ROOT/checks/tiled-gemm/bench.cu"
PROBLEM="$ROOT/problems/tiled-gemm"

[[ -f "$SIF" ]]    || { echo "ERROR: no sif at $SIF (set AGENT_SIF=...)" >&2; exit 1; }
[[ -f "$BENCH" ]]  || { echo "ERROR: missing $BENCH" >&2; exit 1; }

echo "== sif: $SIF"
echo "== GPU visible to the container?"
if ! apptainer exec --nv "$SIF" nvidia-smi -L; then
    echo "ERROR: no GPU visible — run this INSIDE an A40 allocation, e.g.:" >&2
    echo "  srun --gres=gpu:a40:1 --time=00:20:00 --partition=a40 --pty bash scripts/smoke_perf.sh" >&2
    exit 1
fi
echo "== nsys present in image?"
apptainer exec "$SIF" bash -lc 'command -v nsys && nsys --version | head -1 || echo "  nsys ABSENT — check will fall back to cudaEvent timing"'

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
cp -a "$PROBLEM"/. "$WORK_DIR"/

# Stand-in solution.json (what an agent would write): build + run the starter.
cat > "$WORK_DIR/solution.json" <<'JSON'
{ "build": ["make"], "run": ["./gemm -c"], "variant_name": "tiled-gemm" }
JSON

echo "== running measure phase (build + run + perf grading) =="
apptainer exec --nv \
    --env "EVAL_AGENT=pi" \
    --env "EVAL_MODEL=smoke" \
    --env "EVAL_ROUND=1" \
    --env "EVAL_SIZE=4096" \
    --env "EVAL_VARIANT=tiled-gemm" \
    --bind "$WORK_DIR:/problem" \
    --bind "$MEASURE:/measure.sh:ro" \
    --bind "$BENCH:/grade/bench.cu:ro" \
    "$SIF" /bin/bash -l /measure.sh /problem

EVAL="$WORK_DIR/.eval"
echo
echo "================ .eval/perf.log ================"
cat "$EVAL/perf.log" 2>/dev/null || echo "(no perf.log — perf grading did not run)"
echo
echo "== nsys kernel CSVs (if nsys present) =="
for f in kern_agent.csv kern_cublas.csv; do
    echo "--- $f ---"; cat "$EVAL/$f" 2>/dev/null || echo "(absent)"
done

echo
echo "================ host-side check verdict ================"
python3 - "$EVAL" "$ROOT" <<'PY'
import sys, os, importlib.util
eval_dir, root = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(root, "checks"))
sys.path.insert(0, os.path.join(root, "checks", "tiled-gemm"))
spec = importlib.util.spec_from_file_location(
    "check", os.path.join(root, "checks", "tiled-gemm", "check.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
v = m.Check(eval_dir).evaluate()
for s in v.stages:
    print(f"  [{'PASS' if s.passed else 'FAIL'}] {s.name:6} {s.detail}")
print("  overall:", "PASS" if v.passed else "FAIL")
PY