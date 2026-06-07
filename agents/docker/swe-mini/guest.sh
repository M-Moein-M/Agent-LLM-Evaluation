#! /bin/bash
# Evaluation harness (swe-mini agent) — agent produces code + solution.json,
# harness replays commands. Mirrors agents/docker/pi/guest.sh; only the agent
# invocation block (section 1) differs — everything from section 2 onward
# (evidence capture) is identical, so the host-side checks stay agent-agnostic.

set -euo pipefail

PROBLEM_DIR=$1
EVAL_AGENT="${EVAL_AGENT:-swe-mini}"
EVAL_MODEL="${EVAL_MODEL:-unknown-model}"
EVAL_ROUND="${EVAL_ROUND:-0}"
EVAL_SIZE="${EVAL_SIZE:-256}"
EVAL_VARIANT="${EVAL_VARIANT:-LolaBunnyGPU}"

DATE_TAG=$(date +%Y%m%d-%H%M%S)

cd "$PROBLEM_DIR"
rm -rf .git/
rm -rf .github/

# ---------------------------------------------------------------------------
# 1. Run the agent (mini-SWE-agent / litellm against our OpenAI-compatible
#    gateway). The same models.json pi uses is bind-mounted at /models.json
#    and is the single source of the baseUrl + apiKey. mini just runs the bash
#    the model emits, so it writes solution.json exactly as the prompt asks.
# ---------------------------------------------------------------------------
PROMPT=$(cat PROMPT.md)
rm PROMPT.md

tree -a

# Skip mini's first-run interactive setup wizard (no TTY in the container) —
# we supply the model + key + base URL via env and -c overrides below.
export MSWEA_CONFIGURED=true
export OPENAI_API_KEY="$(jq -r '.providers.local.apiKey' /models.json)"
API_BASE="$(jq -r '.providers.local.baseUrl' /models.json)"

# mini's default per-command timeout is 30s — far too short for CUDA builds —
# so bump environment.timeout. cwd is set to the problem dir so each (stateless)
# command starts there. cost_tracking=ignore_errors because our gateway models
# aren't in litellm's price registry. `|| true` keeps the empty/failed-agent
# path working under set -euo pipefail (falls through to the FAIL writer below).
mini -y --exit-immediately \
    -m "openai/$EVAL_MODEL" \
    -c /mini.yaml \
    -c "model.model_kwargs.api_base=$API_BASE" \
    -c "model.model_kwargs.custom_llm_provider=openai" \
    -c "model.cost_tracking=ignore_errors" \
    -c "environment.cwd=$PROBLEM_DIR" \
    -c "environment.timeout=600" \
    -t "$PROMPT" \
    -o ~/agent.traj.json || true
cp ~/agent.traj.json "$PROBLEM_DIR" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 2. Parse agent's solution.json
# ---------------------------------------------------------------------------
SOLUTION_JSON="$PROBLEM_DIR/solution.json"
EVAL_DIR="$PROBLEM_DIR/.eval"
mkdir -p "$EVAL_DIR"

# ---------------------------------------------------------------------------
# facts.json writers (raw receipts only — no interpretation).
# Defined early so the FAIL early-exits below can call them.
# Pass/fail judgment happens host-side in scripts/check_pass.py.
# ---------------------------------------------------------------------------
write_result() {
	cat > "$EVAL_DIR/facts.json" <<EOF
{
	"model": "${EVAL_MODEL}",
	"agent": "${EVAL_AGENT}",
	"round": ${EVAL_ROUND},
	"date": "${DATE_TAG}",
	"size": ${EVAL_SIZE},
	"variant": "${EVAL_VARIANT}",
	"source": "solution.json",
	"build": {
		"cmd": "${BUILD_CMD}",
		"status": ${BUILD_STATUS},
		"start": ${BUILD_START},
		"end": ${BUILD_END}
	},
	"run": {
		"cmd": "${RUN_CMD}",
		"status": ${RUN_STATUS},
		"start": ${RUN_START},
		"end": ${RUN_END}
	}
}
EOF
}

write_fail_result() {
	local reason="$1"
	cat > "$EVAL_DIR/facts.json" <<EOF
{
	"model": "${EVAL_MODEL}",
	"agent": "${EVAL_AGENT}",
	"round": ${EVAL_ROUND},
	"date": "${DATE_TAG}",
	"size": ${EVAL_SIZE},
	"variant": "${EVAL_VARIANT}",
	"source": "solution.json",
	"failure_reason": "${reason}",
	"build": {
		"status": 1,
		"start": 0,
		"end": 0
	},
	"run": {
		"status": 1,
		"start": 0,
		"end": 0
	}
}
EOF
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
# 3. GPU evidence
# ---------------------------------------------------------------------------
GPU_INFO="$EVAL_DIR/gpu_info.txt"
GPU_USAGE="$EVAL_DIR/gpu_usage.csv"

if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi -L > "$GPU_INFO"
else
	echo "nvidia-smi not found" > "$GPU_INFO"
fi

# ---------------------------------------------------------------------------
# 4. Replay build command
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
# 5. Replay run command (with high-resolution GPU sampling)
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
# 6. Write facts.json
# ---------------------------------------------------------------------------
write_result
