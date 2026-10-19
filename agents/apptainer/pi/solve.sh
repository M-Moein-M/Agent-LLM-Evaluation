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
#
# End watch (2026-09-24, same as kimi/solve.sh): if pi is still alive END_GRACE_S
# after its final `agent_end` event, it is ended. The watch reads pi's stdout
# through a FIFO, so text the agent writes into agent.jsonl cannot trigger it.
# Processes left behind die with the container (host.sh runs apptainer exec --pid).
END_GRACE_S="${EVAL_END_GRACE_S:-60}"
end_watch() {  # $1 agent pid, $2 jq condition on the agent's final event; stdin = agent stdout
	jq -R --unbuffered -c "fromjson? | objects | select($2) | 1" 2>/dev/null | {
		if read -r _; then
			local i=0
			while kill -0 "$1" 2>/dev/null && (( i++ < END_GRACE_S )); do sleep 1; done
			if kill -0 "$1" 2>/dev/null; then
				echo "$(date +%FT%T) agent alive ${END_GRACE_S}s after its final event, ending it" \
					>> "$PROBLEM_DIR/agent.end_watch.log"
				kill -TERM "$1" 2>/dev/null || true
				sleep 10
				kill -KILL "$1" 2>/dev/null || true
			fi
		fi
		cat > /dev/null
	}
}
FIFO="$(mktemp -u /tmp/agent-stream.XXXXXX)"
mkfifo "$FIFO"
pi -p --mode json --step-limit "$EVAL_STEP_LIMIT" --model "$EVAL_MODEL" "$PROMPT" \
		> "$FIFO" 2> "$PROBLEM_DIR/agent.stderr.log" &
AGENT_PID=$!
{ tee -p >(end_watch "$AGENT_PID" '.type == "agent_end"') < "$FIFO" & echo $! > "$FIFO.tee"; wait; } \
	| { grep --line-buffered -Ev '"type":"(message_update|tool_execution_update)"' || true; } \
	> "$PROBLEM_DIR/agent.jsonl" &
READER_PID=$!
# pi's exit code stays the script's exit code, except when the end watch ended it.
rc=0
wait "$AGENT_PID" || rc=$?
# A leftover child can hold pi's stdout open, so tee may never see EOF. Let it
# drain for up to 5 s, then stop it.
for _ in 1 2 3 4 5; do kill -0 "$READER_PID" 2>/dev/null || break; sleep 1; done
kill "$(cat "$FIFO.tee")" 2>/dev/null || true
wait "$READER_PID" || true
rm -f "$FIFO" "$FIFO.tee"
[ "$rc" -ne 0 ] && [ -f "$PROBLEM_DIR/agent.end_watch.log" ] && rc=0
exit "$rc"
