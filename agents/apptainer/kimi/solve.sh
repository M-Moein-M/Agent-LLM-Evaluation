#! /bin/bash
# Phase 1 (SOLVE) — kimi (Kimi Code CLI) agent. Mirror of
# agents/apptainer/claude/solve.sh; only the agent-invocation block differs.
# Runs Kimi Code headless against our OpenAI-compatible gateway to produce
# code + solution.json + agent.stream.jsonl.
#
# GPU-contention-tolerant: the agent's own compile/run iterations only need
# correctness feedback, so several solve containers can share the GPU. The
# graded measurement happens later, isolated, in measure.sh.

set -euo pipefail

PROBLEM_DIR=$1
EVAL_MODEL="${EVAL_MODEL:-unknown-model}"
EVAL_MAX_TURNS="${EVAL_MAX_TURNS:-120}"   # enforceable runaway cap (cf. swe-mini step_limit, claude --max-turns)

cd "$PROBLEM_DIR"
rm -rf .git/
rm -rf .github/

# ---------------------------------------------------------------------------
# Point Kimi Code at the gateway. Same models.json the other openai-compat
# agents use is bind-mounted at /models.json and is the single source of the
# baseUrl + apiKey. Unlike claude (Anthropic protocol, needs /v1 stripped),
# Kimi Code speaks OpenAI-compat NATIVELY, so the gateway baseUrl is used
# AS-IS (/v1 kept).
#
# EVAL_PROVIDER names the providers.<name> block in /models.json (generated from
# endpoints.yaml by scripts/lib/resolve_endpoint.py) that supplies baseUrl + apiKey.
EVAL_PROVIDER="${EVAL_PROVIDER:?EVAL_PROVIDER is not set (run via scripts/run_eval.sh)}"
KEY="$(jq -r ".providers.${EVAL_PROVIDER}.apiKey" /models.json)"
BASE="$(jq -r ".providers.${EVAL_PROVIDER}.baseUrl" /models.json)"

# Model config via the KIMI_MODEL_* env channel (the one credential family Kimi
# Code DOES read from the shell — see AGENT.md). Setting KIMI_MODEL_NAME makes
# the CLI synthesize an in-memory provider + model alias and use it as the
# default_model, so there is no config.toml provider block and no -m flag.
export KIMI_MODEL_NAME="$EVAL_MODEL"
export KIMI_MODEL_API_KEY="$KEY"
export KIMI_MODEL_PROVIDER_TYPE="openai"
export KIMI_MODEL_BASE_URL="$BASE"

# Kimi derives max_completion_tokens as min(context, 131072) with NO hard cap
# by default, so it requests 131072 on every call — the gateway 429s
# ("tokens" rate limit, 100k/min) and some served models 400 ("max completion
# tokens" ceiling below that). KIMI_MODEL_MAX_COMPLETION_TOKENS is read ahead
# of that fallback and clamps every request. 64000 mirrors claude's working
# value on this gateway (see AGENT.md's claude token-budget section).
# Unified limits (2026-09-16): no output cap. <=0 disables the completion budget
# (kimi-code utils/completion-budget.ts), so no max_completion_tokens is sent.
export KIMI_MODEL_MAX_COMPLETION_TOKENS="${EVAL_MAX_OUTPUT_TOKENS:-0}"
# Context size above any server window so proactive compaction never fires; the server
# window is the only limit. The patched binary (host.sh) ends the run on a server
# context-overflow 400 instead of compacting, like swe-mini.
export KIMI_MODEL_MAX_CONTEXT_SIZE="${EVAL_KIMI_MAX_CONTEXT_SIZE:-500000}"

# Enforceable iteration cap for the whole run (single turn in -p mode). Takes
# priority over any config.toml value, so no [loop_control] block needs to be
# written. Exceeding it raises LOOP_MAX_STEPS_EXCEEDED — the analog of running
# out of steps.
export KIMI_LOOP_MAX_STEPS_PER_TURN="$EVAL_MAX_TURNS"

# Cross-agent fairness: disable Kimi's micro-compaction, the same policy claude
# runs under (DISABLE_AUTO_COMPACT=1). Left at its default (on), kimi silently
# summarizes older turns when history approaches the context limit and keeps
# working, while pi/swe-mini have no compaction at all and simply die — observed
# 2026-08-10 in job 3973505, where swe-mini hit "Input length (216093 tokens)
# exceeds the maximum allowed length" 11 times (one propagating as an unhandled
# OpenAIError that killed the run) against zero for kimi on the same model. With
# compaction off, a run that exhausts context fails on context exhaustion — the
# analog of the others running out of steps — rather than summarizing to
# continue. Set EVAL_KIMI_COMPACTION=1 to restore the CLI default.
if [ "${EVAL_KIMI_COMPACTION:-0}" != "1" ]; then
    export KIMI_CODE_EXPERIMENTAL_MICRO_COMPACTION=false
fi

# Kimi CLI's own default thinking_effort ("medium") is not universally accepted —
# some vLLM-hosted models (e.g. moonshotai/Kimi-K3) hard-500 on it
# ("Unsupported thinking_effort='medium'; supported values are
# ['high','low','max']"), confirmed against the kimi3 endpoint 2026-07-30. Only
# set the override when asked (EVAL_THINKING_EFFORT), so providers/models that
# already work with the CLI default (e.g. Qwen3.6 — see AGENT.md) are unaffected.
if [ -n "${EVAL_THINKING_EFFORT:-}" ]; then
    export KIMI_MODEL_THINKING_EFFORT="$EVAL_THINKING_EFFORT"
fi

# Headless hygiene: with a custom gateway we never want Kimi Code phoning home
# (telemetry/auto-update) or blocking on a network round-trip mid-run.
export KIMI_DISABLE_TELEMETRY=1
export KIMI_CODE_NO_AUTO_UPDATE=1

# Anti-hang: -p mode's config defaults (applyPrintModeConfigDefaults, per the
# CLI source) zero out background.bash_task_timeout_s (backgrounded Bash
# commands get NO timeout) and leave print_background_mode at its 'steer'
# fallback, which keeps the driver alive re-engaging the agent for as long as
# any background task is outstanding (bounded only by print_wait_ceiling_s /
# print_max_turns, both effectively unbounded by default). Confirmed in job
# 3998608: a solve process kept issuing successful completions for 30+ min
# after its agent.stream.jsonl stopped growing, and only stopped when killed.
# print_background_mode = "exit" makes -p return as soon as the main turn
# completes, regardless of outstanding background tasks. KIMI_CODE_HOME is
# scoped per-problem-dir so concurrent solve containers (EVAL_SOLVE_PAR) never
# share/race on the same config.toml.
export KIMI_CODE_HOME="$PROBLEM_DIR/.kimi-code-home"
mkdir -p "$KIMI_CODE_HOME"
cat > "$KIMI_CODE_HOME/config.toml" <<'EOF'
[task]
bash_auto_background_on_timeout = false
bash_task_timeout_s = 600
print_background_mode = "exit"
EOF

# ---------------------------------------------------------------------------
# Run the agent. The prompt is the problem's PROMPT.md (which carries the
# solution.json contract); remove it afterward for a clean diff, exactly as
# claude/swe-mini do. -p runs a single prompt non-interactively with AUTO
# permission mode already on (no --yolo/--auto needed — -p rejects them
# outright). --output-format stream-json emits one JSON event per line ->
# agent.stream.jsonl. `|| true` keeps the empty/failed-agent path working
# under `set -e` (measure.sh writes the FAIL facts later).
# ---------------------------------------------------------------------------
PROMPT=$(cat PROMPT.md)
rm PROMPT.md

tree -a

# End watch (2026-09-24): kimi stays alive after its final answer while a background
# task runs (the print_background_mode setting above does not stop it). Seen in
# RabbitCT-LolaBunny round 8, where a GPU test program kept kimi alive 3h after the
# session.resume_hint line. The watch
# reads kimi's stdout through a FIFO, so text the agent writes into
# agent.stream.jsonl cannot trigger it. Processes left behind die with the
# container (host.sh runs apptainer exec --pid).
# 2026-09-25: the trigger is kimi's first reply with no tool call, because kimi writes
# session.resume_hint only after its background tasks end (round 5 hung 3h45m on a
# backgrounded `find /` and never wrote it). kimi may still work for END_GRACE_S after
# that reply: a background-task notification can wake it, and in 4 of 381 old rounds
# it did real work then (waited on a baseline run, wrote solution.json). If it is still
# alive after that, it is ended and agent.end_watch.log records it. That file marks
# the round as a kimi failure (hung after its final answer) for the report.
END_GRACE_S="${EVAL_END_GRACE_S:-600}"
end_watch() {  # $1 agent pid, $2 jq condition on the agent's final event; stdin = agent stdout
    jq -R --unbuffered -c "fromjson? | objects | select($2) | 1" 2>/dev/null | {
        if read -r _; then
            local i=0
            while kill -0 "$1" 2>/dev/null && (( i++ < END_GRACE_S )); do sleep 1; done
            if kill -0 "$1" 2>/dev/null; then
                echo "$(date +%FT%T) hung_after_final: agent alive ${END_GRACE_S}s after its first reply with no tool call, ending it" \
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
kimi -p "$PROMPT" \
    --output-format stream-json \
    > "$FIFO" 2> "$PROBLEM_DIR/agent.stderr.log" &
AGENT_PID=$!
{ tee -p >(end_watch "$AGENT_PID" '.role == "assistant" and ((.tool_calls // []) | length) == 0') \
    < "$FIFO" & echo $! > "$FIFO.tee"; wait; } > "$PROBLEM_DIR/agent.stream.jsonl" &
READER_PID=$!
wait "$AGENT_PID" || true
# A leftover child can hold kimi's stdout open, so tee may never see EOF. Let it
# drain for up to 5 s, then stop it.
for _ in 1 2 3 4 5; do kill -0 "$READER_PID" 2>/dev/null || break; sleep 1; done
kill "$(cat "$FIFO.tee")" 2>/dev/null || true
wait "$READER_PID" || true
rm -f "$FIFO" "$FIFO.tee"

# Keep the session wire logs (per-call usage.record, compaction events) in the trace.
if [ -d "$KIMI_CODE_HOME/sessions" ]; then
    mkdir -p "$PROBLEM_DIR/agent.kimi-wire"
    find "$KIMI_CODE_HOME/sessions" -name wire.jsonl | while read -r w; do
        cp "$w" "$PROBLEM_DIR/agent.kimi-wire/$(basename "$(dirname "$w")").wire.jsonl"
    done
fi
rm -rf "$KIMI_CODE_HOME"
