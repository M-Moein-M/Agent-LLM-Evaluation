#! /bin/bash
# Phase 2 (MEASURE) — swe-mini agent. Identical in spirit to
# agents/docker/pi/measure.sh (and agent-agnostic): replays the agent's
# solution.json build/run with GPU sampling and writes raw evidence to .eval/.
#
# Runs SERIALLY (one container per GPU) so wall-clock timing and the 250ms
# gpu_usage.csv sampling reflect THIS run alone. Interprets nothing — pass/fail
# judgment happens host-side in scripts/check_pass.py.

set -euo pipefail

PROBLEM_DIR=$1
EVAL_AGENT="${EVAL_AGENT:-swe-mini}"
EVAL_MODEL="${EVAL_MODEL:-unknown-model}"
EVAL_ROUND="${EVAL_ROUND:-0}"
EVAL_SIZE="${EVAL_SIZE:-256}"
EVAL_VARIANT="${EVAL_VARIANT:-LolaBunnyGPU}"

DATE_TAG=$(date +%Y%m%d-%H%M%S)

cd "$PROBLEM_DIR"

# ---------------------------------------------------------------------------
# Parse agent's solution.json
# ---------------------------------------------------------------------------
SOLUTION_JSON="$PROBLEM_DIR/solution.json"
EVAL_DIR="$PROBLEM_DIR/.eval"
mkdir -p "$EVAL_DIR"

# ---------------------------------------------------------------------------
# facts.json writers (raw receipts only — no interpretation).
# Defined early so the FAIL early-exits below can call them.
# Pass/fail judgment happens host-side in scripts/check_pass.py.
# ---------------------------------------------------------------------------
# Build facts.json with jq so build/run COMMANDS (which can contain quotes,
# e.g. LolaOMP's CUDA_ARCH="-gencode=...") are JSON-escaped. Raw heredoc
# interpolation produced invalid JSON for such commands, which crashed the
# host-side check. jq is present in the image.
write_result() {
	jq -n \
		--arg model "$EVAL_MODEL" \
		--arg agent "$EVAL_AGENT" \
		--argjson round "$EVAL_ROUND" \
		--arg date "$DATE_TAG" \
		--argjson size "$EVAL_SIZE" \
		--arg variant "$EVAL_VARIANT" \
		--arg build_cmd "$BUILD_CMD" \
		--argjson build_status "$BUILD_STATUS" \
		--argjson build_start "$BUILD_START" \
		--argjson build_end "$BUILD_END" \
		--arg run_cmd "$RUN_CMD" \
		--argjson run_status "$RUN_STATUS" \
		--argjson run_start "$RUN_START" \
		--argjson run_end "$RUN_END" \
		'{model:$model, agent:$agent, round:$round, date:$date, size:$size,
		  variant:$variant, source:"solution.json",
		  build:{cmd:$build_cmd, status:$build_status, start:$build_start, end:$build_end},
		  run:{cmd:$run_cmd, status:$run_status, start:$run_start, end:$run_end}}' \
		> "$EVAL_DIR/facts.json"
}

write_fail_result() {
	local reason="$1"
	jq -n \
		--arg model "$EVAL_MODEL" \
		--arg agent "$EVAL_AGENT" \
		--argjson round "$EVAL_ROUND" \
		--arg date "$DATE_TAG" \
		--argjson size "$EVAL_SIZE" \
		--arg variant "$EVAL_VARIANT" \
		--arg reason "$reason" \
		'{model:$model, agent:$agent, round:$round, date:$date, size:$size,
		  variant:$variant, source:"solution.json", failure_reason:$reason,
		  build:{status:1, start:0, end:0},
		  run:{status:1, start:0, end:0}}' \
		> "$EVAL_DIR/facts.json"
}

if [ ! -f "$SOLUTION_JSON" ]; then
	echo "WARN: No solution.json — writing FAIL result" >&2
	write_fail_result "No solution.json produced"
	exit 0
fi

BUILD_CMD=$(jq -r '.build[0] // empty' "$SOLUTION_JSON" 2>/dev/null) || true
RUN_CMD=$(jq -r '.run[0] // empty' "$SOLUTION_JSON" 2>/dev/null) || true
VARIANT=$(jq -r '.variant_name // empty' "$SOLUTION_JSON" 2>/dev/null) || true

if [ -z "$BUILD_CMD" ] || [ -z "$RUN_CMD" ]; then
	echo "WARN: solution.json missing build/run — writing FAIL result" >&2
	write_fail_result "Incomplete solution.json (missing build/run)"
	exit 0
fi

if [ -n "$VARIANT" ]; then
	EVAL_VARIANT="$VARIANT"
fi

echo "Harness replaying: build='$BUILD_CMD' run='$RUN_CMD' variant='$EVAL_VARIANT'"

# ---------------------------------------------------------------------------
# GPU evidence
# ---------------------------------------------------------------------------
GPU_INFO="$EVAL_DIR/gpu_info.txt"
GPU_USAGE="$EVAL_DIR/gpu_usage.csv"

if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi -L > "$GPU_INFO"
else
	echo "nvidia-smi not found" > "$GPU_INFO"
fi

# ---------------------------------------------------------------------------
# Replay build command
# ---------------------------------------------------------------------------
BUILD_LOG="$EVAL_DIR/build.log"
BUILD_START=$(date +%s)
set +e
bash -lc "$BUILD_CMD" > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
BUILD_END=$(date +%s)

RUN_STATUS=1
RUN_START=0
RUN_END=0

# ---------------------------------------------------------------------------
# Replay run command (with high-resolution GPU sampling)
# ---------------------------------------------------------------------------
if [ "$BUILD_STATUS" -eq 0 ]; then
	RUN_LOG="$EVAL_DIR/run.log"
	RUN_START=$(date +%s)

	# Single background nvidia-smi writer at 250ms resolution — gives ~hundreds of
	# samples over the run. The host-side check parses utilization from this csv.
	SMI_PID=""
	if command -v nvidia-smi >/dev/null 2>&1; then
		nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
			--format=csv,nounits -lms 250 -f "$GPU_USAGE" &
		SMI_PID=$!
	fi

	bash -lc "$RUN_CMD" > "$RUN_LOG" 2>&1
	RUN_STATUS=$?
	RUN_END=$(date +%s)

	if [ -n "$SMI_PID" ]; then
		kill "$SMI_PID" 2>/dev/null || true
		wait "$SMI_PID" 2>/dev/null || true
	fi
fi

# ---------------------------------------------------------------------------
# Write facts.json
# ---------------------------------------------------------------------------
write_result
