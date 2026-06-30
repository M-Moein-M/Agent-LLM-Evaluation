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
pi -p --mode json --step-limit "$EVAL_STEP_LIMIT" --model "$EVAL_MODEL" "$PROMPT" 2>/dev/null \
	| { grep --line-buffered -Ev '"type":"(message_update|tool_execution_update)"' || true; } \
	| tee ~/agent.jsonl
cp ~/agent.jsonl "$PROBLEM_DIR"
