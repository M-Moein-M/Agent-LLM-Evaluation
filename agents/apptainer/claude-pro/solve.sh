#! /bin/bash
# Phase 1 (SOLVE) — claude-pro agent: Claude Code on a PERSONAL CLAUDE
# SUBSCRIPTION (OAuth), talking to api.anthropic.com. Fork of
# agents/apptainer/claude/solve.sh; only the auth/model block differs — that one
# points Claude Code at the gateway with a Bearer API key and pins every tier
# slot to the single served model. Here the real Anthropic API serves all tiers,
# so there is nothing to pin and no gateway URL to set.
#
# GPU-contention-tolerant: the agent's own compile/run iterations only need
# correctness feedback, so several solve containers can share the GPU. The
# graded measurement happens later, isolated, in measure.sh.

set -euo pipefail

PROBLEM_DIR=$1
EVAL_MODEL="${EVAL_MODEL:-claude-opus-5}"
EVAL_MAX_TURNS="${EVAL_MAX_TURNS:-120}"   # enforceable runaway cap (cf. swe-mini step_limit=120)

# Reasoning effort, passed as `claude --effort <level>` (low|medium|high|xhigh|
# max; verified present in the pinned CLI 2.1.209). Session-scoped, overrides the
# effortLevel setting without persisting. Set empty to fall back to the CLI default.
#
# CAUTION — higher effort means longer reasoning per turn, and with
# DISABLE_AUTO_COMPACT=1 every retained thinking block counts against the window.
# Opus already burns ~3x the context per turn that the gateway's DeepSeek does
# (1883-4130 vs 925-1450 tok/turn on the 20260728 waves). Compare ctx/turn between
# waves before reading any effort effect as a capability result.
EVAL_EFFORT="${EVAL_EFFORT:-high}"

cd "$PROBLEM_DIR"
rm -rf .git/
rm -rf .github/

# ---------------------------------------------------------------------------
# Auth: subscription OAuth token, bind-mounted read-only at /oauth_token by
# host.sh (minted once with `claude setup-token`, see this dir's README.md).
# CLAUDE_CODE_OAUTH_TOKEN is the headless equivalent of an interactive
# `claude login` — no browser, no ~/.claude/.credentials.json needed.
#
# Any ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY inherited
# from the environment would silently re-route us back to the gateway (or make
# Claude Code prefer API-key billing over the subscription), so clear them.
# ---------------------------------------------------------------------------
unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY

if [ ! -s /oauth_token ]; then
    echo "ERROR: /oauth_token is missing or empty — mint one with 'claude setup-token' (see agents/apptainer/claude-pro/README.md)" >&2
    exit 1
fi
CLAUDE_CODE_OAUTH_TOKEN="$(tr -d '[:space:]' < /oauth_token)"
export CLAUDE_CODE_OAUTH_TOKEN

# Real Anthropic model id (claude-opus-4-8, claude-sonnet-5, ...). Unlike the
# gateway agent we do NOT pin the SONNET/OPUS/HAIKU tier slots: the real API
# serves every tier, so Claude Code's internal small/fast calls route correctly
# on their own defaults (and pinning them to Opus would burn subscription quota
# on trivial calls).
export ANTHROPIC_MODEL="$EVAL_MODEL"

# Per-response output-token budget. The gateway agent needs an explicit 64000
# here to thread its 262144 shared-window needle; on the real API the default is
# already sane, so this stays UNSET unless overridden per run.
if [ -n "${EVAL_MAX_OUTPUT_TOKENS:-}" ]; then
    export CLAUDE_CODE_MAX_OUTPUT_TOKENS="$EVAL_MAX_OUTPUT_TOKENS"
fi

# Extended thinking. Unlike the gateway path (where this is a documented no-op —
# Claude Code doesn't emit `thinking.budget_tokens` for models outside its
# registry), a real Claude model IS recognized, so this lever actually works here.
if [ -n "${EVAL_MAX_THINKING_TOKENS:-}" ]; then
    export MAX_THINKING_TOKENS="$EVAL_MAX_THINKING_TOKENS"
fi

# Headless hygiene: no auto-update (the image's binary is the pinned one), no
# onboarding wizard, no error reporting. NB: unlike the gateway agent we do NOT
# set CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC — on a subscription that path also
# carries usage/quota bookkeeping to Anthropic.
export DISABLE_AUTOUPDATER=1
export DISABLE_TELEMETRY=1
export DISABLE_ERROR_REPORTING=1

# Disable automatic context compaction — CROSS-AGENT FAIRNESS: the other agents
# (pi, swe-mini, claude) have no compaction, so claude-pro must not get a free
# context extension the others lack. A run that grows past the window fails on
# context exhaustion (comparable to the others running out of steps).
export DISABLE_AUTO_COMPACT=1

# The context window is NOT set here — the model's native window applies (1M on a
# current Opus subscription), so total tokens per round run as high as the model
# allows. Nothing in this repo may set CLAUDE_CODE_AUTO_COMPACT_WINDOW: it is a
# token count, not a compaction toggle, and any value clamps the window. The
# 20260728 wave lost all 5 rounds that way when a host ~/.bashrc export leaked in;
# --cleanenv in host.sh now blocks that (full story in AGENT.md).

# Skip the first-run onboarding wizard (no TTY in the container). HOME here is
# the per-job staging home host.sh created — nothing lands in the real home.
mkdir -p ~/.claude
printf '{"hasCompletedOnboarding": true}\n' > ~/.claude.json

# ---------------------------------------------------------------------------
# Run the agent. Identical to the gateway agent: prompt is the problem's
# PROMPT.md (which carries the solution.json contract), removed afterward for a
# clean diff. --output-format stream-json --verbose emits one JSON event per line
# → agent.stream.jsonl, which trace_stats.py parses for tokens/turns/steps.
# `|| true` keeps the empty/failed-agent path working under `set -e` (measure.sh
# writes the FAIL facts later).
# ---------------------------------------------------------------------------
PROMPT=$(cat PROMPT.md)
rm PROMPT.md

tree -a

EFFORT_FLAG=()
[ -n "$EVAL_EFFORT" ] && EFFORT_FLAG=(--effort "$EVAL_EFFORT")

claude -p "$PROMPT" \
    --model "$EVAL_MODEL" \
    --max-turns "$EVAL_MAX_TURNS" \
    "${EFFORT_FLAG[@]}" \
    --permission-mode bypassPermissions \
    --allowedTools "Read,Edit,Write,Bash" \
    --output-format stream-json --verbose \
    > "$PROBLEM_DIR/agent.stream.jsonl" 2> "$PROBLEM_DIR/agent.stderr.log" || true
