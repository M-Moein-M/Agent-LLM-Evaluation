#! /bin/bash
# Phase 1 (SOLVE): run the agent to produce code + solution.json.
#
# This is the GPU-CONTENTION-TOLERANT half of the old guest.sh. The agent may
# compile and run its code many times while iterating, but those inner runs are
# only *correctness* feedback for the agent — not the graded measurement. CUDA
# time-slices kernels across processes, so several solve containers can share
# one GPU concurrently and still get correct results (just slower). The graded
# timing + gpu_usage.csv is produced later, in isolation, by measure.sh.
#
# Output left in the workdir: solution.json (written by the agent), modified
# sources, and agent.jsonl (the agent's own trace). No .eval/ is written here.

set -euo pipefail

PROBLEM_DIR=$1
EVAL_MODEL="${EVAL_MODEL:-unknown-model}"
# Hard cap on assistant turns per prompt (pi fork's --step-limit flag). Bounds
# runaway solve loops so a model that never converges can't burn the whole job.
EVAL_STEP_LIMIT="${EVAL_STEP_LIMIT:-120}"

mkdir -p /root/.pi/agent
cd "$PROBLEM_DIR"
rm -rf .git/
rm -rf .github/

# ---------------------------------------------------------------------------
# Run the agent (code creation + self-debugging loop)
# ---------------------------------------------------------------------------
PROMPT=$(cat PROMPT.md)
rm PROMPT.md

tree -a
tree /root/.pi -a

pi --list-models
# Drop the high-frequency streaming delta events before persisting the trace.
# `message_update` / `tool_execution_update` re-serialize the *entire* accumulated
# message on every token, which is quadratic and was ~99.7% of a 272MB trace.
# The final content is preserved losslessly in message_end / tool_execution_end /
# agent_end. `|| true` keeps the empty-output fail path working under pipefail.
#
# NOTE: no wall-clock cap on the solve. A self-hosted endpoint can be legitimately
# slow (queued/loaded), so a coarse timeout here risks killing a run that is still
# making progress. --step-limit bounds the NUMBER of turns; a truly hung tool call
# is handled out-of-band (kill the leaf proc) per [[pi-solve-no-tool-timeout-hang]].
# The graded replay in measure.sh IS capped, since Phase 2 is serial.
#
# The trace is written straight into $PROBLEM_DIR, not staged in $HOME and copied
# after. Under pipefail a failed write on the shared job stdout aborts the script,
# and the old `tee ~/agent.jsonl` + trailing `cp` lost the whole trace when that
# happened -- the round still measured, because the sources were intact, so it
# looked complete with no log. Writing in place means the file survives any exit.
# pi's stderr is kept for the same reason: 2>/dev/null left nothing to diagnose.
pi -p --mode json --step-limit "$EVAL_STEP_LIMIT" --model "$EVAL_MODEL" "$PROMPT" \
		2> "$PROBLEM_DIR/agent.stderr.log" \
	| { grep --line-buffered -Ev '"type":"(message_update|tool_execution_update)"' || true; } \
	> "$PROBLEM_DIR/agent.jsonl"
