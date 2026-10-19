#! /bin/bash
# Phase 2 (MEASURE): replay the agent's build/run and capture graded evidence.
#
# ONE canonical copy, shared by every agent — every host.sh
# binds this same file to /measure.sh. It is agent-agnostic by construction: it
# reads solution.json, replays it, and records raw receipts.
#
# This is the CONTENTION-INTOLERANT half of the old guest.sh. It must run
# SERIALLY, so the wall-clock timing and the 250ms gpu_usage.csv sampling
# reflect THIS run alone — co-located work would corrupt exactly the numbers the
# host-side checks grade on. That applies to CPU problems too: a CPU kernel
# timed next to three other builds measures the neighbours, not the kernel.
#
# Runs on GPU and CPU-only nodes alike. The GPU evidence below is best-effort:
# where there is no nvidia-smi there is no gpu_usage.csv, and the CPU problems'
# checks do not ask for one.
#
# Input: a workdir already populated by solve.sh (solution.json + sources).
# Output: .eval/ (facts.json, build.log, run.log, host_info.txt, and on GPU
# nodes gpu_info.txt + gpu_usage.csv).
# Interprets nothing — pass/fail judgment happens host-side in check_pass.py.

set -euo pipefail

PROBLEM_DIR=$1
EVAL_AGENT="${EVAL_AGENT:-pi}"
EVAL_MODEL="${EVAL_MODEL:-unknown-model}"
EVAL_ROUND="${EVAL_ROUND:-0}"
EVAL_SIZE="${EVAL_SIZE:-256}"
EVAL_VARIANT="${EVAL_VARIANT:-LolaBunnyGPU}"

DATE_TAG=$(date +%Y%m%d-%H%M%S)

# Which machine produced this measurement. Correctness (the DIGEST) is
# node-independent, but the ranking metric is wall time on a specific CPU, so a
# round is only comparable to another round from the same node. Record it in
# every trace rather than relying on whoever submitted the job to remember.
EVAL_NODE="${SLURMD_NODENAME:-$(hostname -s 2>/dev/null || hostname)}"

# EVAL_REQUIRE_GPU=1 (Phase 6 worker): without a visible GPU the replay would fail
# and be recorded as a real run_error. Exit before writing any facts instead, so
# the round stays unmeasured and is measured again later.
if [[ "${EVAL_REQUIRE_GPU:-0}" == "1" ]] && ! nvidia-smi -L 2>/dev/null | grep -q "^GPU"; then
	echo "ERROR: EVAL_REQUIRE_GPU=1 but no GPU is visible in the container, no facts written" >&2
	exit 3
fi

cd "$PROBLEM_DIR"

# ---------------------------------------------------------------------------
# Parse agent's solution.json
# ---------------------------------------------------------------------------
SOLUTION_JSON="$PROBLEM_DIR/solution.json"
EVAL_DIR="$PROBLEM_DIR/.eval"
mkdir -p "$EVAL_DIR"

# ---------------------------------------------------------------------------
# Optional perf grading: harness-owned bench + cuBLAS reference, timed by nsys.
# Runs ONLY when a grade bench is bound in for this problem (host.sh binds
# checks/<problem>/bench.cu -> /grade/bench.cu). Other problems and the
# correctness path above are untouched. Emits raw evidence to .eval/; the
# host-side check computes the agent-vs-cuBLAS kernel-time RATIO — no
# operation-count assumption, self-calibrating to whatever GPU runs it.
#
# The paired run (.eval/perf_paired.log) times cuBLAS before and after the
# agent kernel in one process, so a disturbed measurement shows up as a spread
# in the cuBLAS segments. The host-side check prefers it when present.
# $1 = log that receives the bench build output.
# ---------------------------------------------------------------------------
perf_build() {
	{
		echo "== perf grading: build gemm_bench from harness bench + agent kernel =="
		# Agent kernel sources = every src/*.cu WITHOUT its own int main, so the
		# agent stays free to split the kernel across files; the bench owns main().
		KSRCS=$(grep -L 'int main' src/*.cu 2>/dev/null | tr '\n' ' ')
		echo "agent kernel sources: $KSRCS"
		set -x
		# -arch=sm_86 targets the A40 (Ampere GA102) explicitly — deterministic
		# and independent of build-time GPU detection (override BENCH_ARCH for
		# other GPUs).
		nvcc -O3 -std=c++14 -arch="${BENCH_ARCH:-sm_86}" /grade/bench.cu $KSRCS \
			-I src -o gemm_bench -lcublas
		set +x
	} > "$1" 2>&1
}

perf_paired() {
	# Cap at 20 min: 9 segments of 55 launches. A kernel at 1% of cuBLAS needs
	# about 6 min.
	timeout 1200 ./gemm_bench paired >> "$EVAL_DIR/perf_paired.log" 2>&1 || true
}

# EVAL_PERF_ONLY=1 (scripts/remeasure.sh --perf-only): only the paired run, for
# rounds that are already measured. perf.log, the nsys files and facts.json are
# not touched.
if [[ "${EVAL_PERF_ONLY:-0}" == "1" ]]; then
	if [ -f /grade/bench.cu ] && command -v nvcc >/dev/null 2>&1; then
		rm -f gemm_bench
		# "|| true": a failed bench build must reach the message below.
		perf_build "$EVAL_DIR/perf_paired.log" || true
		if [ -x ./gemm_bench ]; then
			perf_paired
		else
			echo "gemm_bench did not build — no paired perf result" >> "$EVAL_DIR/perf_paired.log"
		fi
	fi
	exit 0
fi

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
		--arg node "$EVAL_NODE" \
		--arg build_cmd "$BUILD_CMD" \
		--argjson build_status "$BUILD_STATUS" \
		--argjson build_start "$BUILD_START" \
		--argjson build_end "$BUILD_END" \
		--arg run_cmd "$RUN_CMD" \
		--argjson run_status "$RUN_STATUS" \
		--argjson run_start "$RUN_START" \
		--argjson run_end "$RUN_END" \
		'{model:$model, agent:$agent, round:$round, date:$date, size:$size,
		  variant:$variant, node:$node, source:"solution.json",
		  build:{cmd:$build_cmd, status:$build_status, start:$build_start, end:$build_end},
		  run:{cmd:$run_cmd, status:$run_status, start:$run_start, end:$run_end}}' \
		> "$EVAL_DIR/facts.json"
}

# $1 is a STABLE SLUG, $2 the prose. The slug is what the host-side reports key
# on (checks/failure_class.py -> the round's fail_class), so it must not be
# reworded casually; the prose is free text for a human reading facts.json.
# Older traces carry only the prose and are mapped back by FACTS_FAILURE_CODE.
write_fail_result() {
	local code="$1"
	local reason="$2"
	jq -n \
		--arg model "$EVAL_MODEL" \
		--arg agent "$EVAL_AGENT" \
		--argjson round "$EVAL_ROUND" \
		--arg date "$DATE_TAG" \
		--argjson size "$EVAL_SIZE" \
		--arg variant "$EVAL_VARIANT" \
		--arg node "$EVAL_NODE" \
		--arg code "$code" \
		--arg reason "$reason" \
		'{model:$model, agent:$agent, round:$round, date:$date, size:$size,
		  variant:$variant, node:$node, source:"solution.json",
		  failure_code:$code, failure_reason:$reason,
		  build:{status:1, start:0, end:0},
		  run:{status:1, start:0, end:0}}' \
		> "$EVAL_DIR/facts.json"
}

if [ ! -f "$SOLUTION_JSON" ]; then
	echo "WARN: No solution.json — writing FAIL result" >&2
	write_fail_result no_solution "No solution.json produced"
	exit 0
fi

BUILD_CMD=$(jq -r '.build[0] // empty' "$SOLUTION_JSON" 2>/dev/null) || true
RUN_CMD=$(jq -r '.run[0] // empty' "$SOLUTION_JSON" 2>/dev/null) || true
VARIANT=$(jq -r '.variant_name // empty' "$SOLUTION_JSON" 2>/dev/null) || true

if [ -z "$BUILD_CMD" ] || [ -z "$RUN_CMD" ]; then
	echo "WARN: solution.json missing build/run — writing FAIL result" >&2
	write_fail_result incomplete_solution "Incomplete solution.json (missing build/run)"
	exit 0
fi

if [ -n "$VARIANT" ]; then
	EVAL_VARIANT="$VARIANT"
fi

echo "Harness replaying: build='$BUILD_CMD' run='$RUN_CMD' variant='$EVAL_VARIANT'"

# ---------------------------------------------------------------------------
# Host evidence. For CPU problems this IS the hardware record: which node, which
# CPU, how many cores, what the topology looked like from inside the container.
# Cheap, and it is the only way to tell after the fact whether two rounds are
# even comparable.
# ---------------------------------------------------------------------------
{
	echo "node: $EVAL_NODE"
	echo "date: $(date -Is)"
	echo "nproc: $(nproc 2>/dev/null || echo '?')"
	echo "--- lscpu ---"
	lscpu 2>/dev/null || echo "lscpu unavailable"
	echo "--- numactl -H ---"
	numactl -H 2>/dev/null || echo "numactl unavailable"
} > "$EVAL_DIR/host_info.txt" 2>&1

# ---------------------------------------------------------------------------
# GPU evidence (best-effort — CPU-only nodes simply have none)
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
# Cap the build (20 min). A clean build of the heaviest problem (RabbitCT) is
# ~1-2 min; this only fires on a wedged build. timeout exit 124 -> BUILD_STATUS
# non-zero -> treated as a build failure (run is skipped) by the logic below.
timeout 1200 bash -lc "$BUILD_CMD" > "$BUILD_LOG" 2>&1
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

	# Cap the graded run (10 min). Phase 2 is SERIAL and has no step-limit, so a
	# non-terminating agent binary (e.g. an infinite-loop ./compact) would block
	# this measure AND every measure after it — the one spot a hang has no escape.
	# A real graded run is seconds; timeout exit 124 -> RUN_STATUS non-zero -> fail.
	timeout 600 bash -lc "$RUN_CMD" > "$RUN_LOG" 2>&1
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

# ---------------------------------------------------------------------------
# Optional perf grading (perf_build / perf_paired are defined near the top).
# ---------------------------------------------------------------------------
if [ -f /grade/bench.cu ] && command -v nvcc >/dev/null 2>&1; then
	PERF_LOG="$EVAL_DIR/perf.log"
	perf_build "$PERF_LOG"
	if [ -x ./gemm_bench ]; then
		if command -v nsys >/dev/null 2>&1; then
			# --capture-range=cudaProfilerApi: the report holds ONLY the timed
			# loop (the bench brackets it with cudaProfilerStart/Stop) — no
			# warmup, no allocation, no H2D copies.
			for m in agent cublas; do
				nsys profile --force-overwrite true \
					--capture-range=cudaProfilerApi --capture-range-end=stop \
					-o "$EVAL_DIR/prof_$m" ./gemm_bench "$m" >> "$PERF_LOG" 2>&1 || true
				nsys stats --report cuda_gpu_kern_sum --format csv \
					"$EVAL_DIR/prof_$m.nsys-rep" > "$EVAL_DIR/kern_$m.csv" \
					2>>"$PERF_LOG" || true
			done
		else
			echo "nsys absent — host check falls back to bench cudaEvent timing" >> "$PERF_LOG"
			./gemm_bench agent  >> "$PERF_LOG" 2>&1 || true
			./gemm_bench cublas >> "$PERF_LOG" 2>&1 || true
		fi
		: > "$EVAL_DIR/perf_paired.log"
		perf_paired
	else
		echo "gemm_bench did not build — no perf score (correctness still graded)" >> "$PERF_LOG"
	fi
fi
