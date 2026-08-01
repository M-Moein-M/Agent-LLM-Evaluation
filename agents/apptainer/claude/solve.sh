#! /bin/bash
# Phase 1 (SOLVE) — claude (Claude Code) agent. Mirror of
# agents/docker/swe-mini/solve.sh; only the agent-invocation block differs.
# Runs Claude Code headless against our gateway (Anthropic protocol) to
# produce code + solution.json + agent.stream.jsonl.
#
# GPU-contention-tolerant: the agent's own compile/run iterations only need
# correctness feedback, so several solve containers can share the GPU. The
# graded measurement happens later, isolated, in measure.sh.

set -euo pipefail

PROBLEM_DIR=$1
EVAL_MODEL="${EVAL_MODEL:-unknown-model}"
EVAL_MAX_TURNS="${EVAL_MAX_TURNS:-120}"   # enforceable runaway cap (cf. swe-mini step_limit=120)

# Reasoning effort, passed as `claude --effort <level>`. Set here so the CLI
# invocation stays IDENTICAL to claude-pro's — the two agents must differ only in
# auth/model, or the gateway-vs-subscription comparison stops being clean.
#
# Verified 2026-07-29: the pinned CLI 2.1.209 ACCEPTS --effort high/xhigh against
# the gateway without error. That is not evidence it does anything — effort is a
# Claude-model reasoning control and the gateway serves non-Anthropic models
# (models.json already declares supportsReasoningEffort:false for this provider),
# so treat it as a no-op here and do NOT attribute any DeepSeek score change to it.
EVAL_EFFORT="${EVAL_EFFORT:-high}"

cd "$PROBLEM_DIR"
rm -rf .git/
rm -rf .github/

# ---------------------------------------------------------------------------
# Point Claude Code at the gateway. Same models.json the openai-compat
# agents use is bind-mounted at /models.json and is the single source of the
# baseUrl + apiKey. Claude Code talks the *Anthropic* protocol (POSTs
# /v1/messages), so the base URL must NOT include the openai-compat /v1
# suffix that models.json carries — strip it. Auth via ANTHROPIC_AUTH_TOKEN
# sends `Authorization: Bearer <key>`, which is what the gateway expects (the
# pre-run curl probes in AGENT.md use the same Bearer scheme).
# ---------------------------------------------------------------------------
# EVAL_PROVIDER selects which providers.<name> block in /models.json supplies the
# baseUrl + apiKey (default "local" = gateway; "llama" = the host endpoint).
EVAL_PROVIDER="${EVAL_PROVIDER:-local}"
KEY="$(jq -r ".providers.${EVAL_PROVIDER}.apiKey" /models.json)"
BASE="$(jq -r ".providers.${EVAL_PROVIDER}.baseUrl" /models.json)"
BASE="${BASE%/v1}"          # https://gateway.example.org/v1 -> .../llmgw

export ANTHROPIC_BASE_URL="$BASE"
export ANTHROPIC_AUTH_TOKEN="$KEY"

# The gateway serves ONE model at a time, and Claude Code routes some internal
# calls to its "small/fast" (HAIKU) tier slot. If any tier slot points at a
# model the gateway isn't currently serving, claude silently fails on that
# tier — so pin ALL tier slots (primary + SONNET/OPUS/HAIKU + the legacy
# small-fast var) to the single served EVAL_MODEL. Override only if every
# targeted model is served at once (not the first-A40-run case).
export ANTHROPIC_MODEL="$EVAL_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$EVAL_MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$EVAL_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$EVAL_MODEL"
export ANTHROPIC_SMALL_FAST_MODEL="$EVAL_MODEL"

# Per-response output-token budget Claude Code requests (the `max_tokens` field
# in each /v1/messages call) — NOT the context window. This budget is RESERVED
# against the model's shared 262144 window (max_model_len = input + output), so
# it must thread a needle between two observed failure modes:
#   * too low (32000 default): a verbose thinking model (Qwen3.6) burns the whole
#     budget on one `thinking` block and stops with stop_reason=max_tokens BEFORE
#     emitting the answer / solution.json.
#   * too high (e.g. 128000): once the conversation's input grows, input+output
#     exceeds 262144 and EVERY call 400s with ContextWindowExceededError — fatal
#     here because auto-compaction is disabled below (no recovery).
# 64000 fits Qwen's ~30K verbose turns and still leaves ~200K for input. Override
# per run with EVAL_MAX_OUTPUT_TOKENS.
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${EVAL_MAX_OUTPUT_TOKENS:-64000}"

# Optional extended-thinking effort. KNOWN INEFFECTIVE FOR THIS GATEWAY MODEL —
# kept only as a plumbed hook. Claude Code is supposed to map MAX_THINKING_TOKENS
# to the Anthropic `thinking.budget_tokens` field, but it only emits that field
# for models it RECOGNIZES as extended-thinking-capable; our custom-named
# Qwen/Qwen3.6 is not in its registry, so Claude Code sends NO thinking field and
# the env var is a no-op. Qwen then free-runs its NATIVE chain-of-thought
# unbounded (the gateway surfaces it as Anthropic "thinking" blocks), measured at
# ~12-34K tokens/turn regardless of this value. Verified: forwarding
# EVAL_MAX_THINKING_TOKENS=4096 still produced 24K-token thinking blocks. The only
# lever that actually bounds Qwen thinking is Qwen-native (`/no_think` /
# chat_template_kwargs.enable_thinking) which Claude Code cannot inject. Net effect
# on gemm: unbounded thinking grows context fast and hits the 262144 wall (fatal
# "Prompt is too long", since auto-compact is off). Left here in case a future
# served model IS recognized by Claude Code.
if [ -n "${EVAL_MAX_THINKING_TOKENS:-}" ]; then
    export MAX_THINKING_TOKENS="$EVAL_MAX_THINKING_TOKENS"
fi

# Headless hygiene: with a custom gateway we never want Claude Code phoning
# home (telemetry/auto-update/error reporting) or blocking on onboarding.
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export DISABLE_AUTOUPDATER=1
export DISABLE_TELEMETRY=1
export DISABLE_ERROR_REPORTING=1

# Omit the attribution block (client version + prompt fingerprint) from the
# system prompt — it changes per call and busts the gateway's prompt cache, so
# dropping it improves cache hit rates when routing through the gateway.
export CLAUDE_CODE_ATTRIBUTION_HEADER=0

# Disable automatic context compaction — for CROSS-AGENT FAIRNESS: the other
# agents (pi, swe-mini) have no compaction, so claude must not get a free context
# extension the others lack. A run that grows past the window therefore fails on
# context exhaustion (comparable to the others running out of steps), rather than
# silently summarizing to keep going. Keep CLAUDE_CODE_MAX_OUTPUT_TOKENS modest
# (above) so this wall isn't hit prematurely by an oversized output reservation.
export DISABLE_AUTO_COMPACT=1
# Skip the first-run onboarding wizard (no TTY in the container).
mkdir -p ~/.claude
printf '{"hasCompletedOnboarding": true}\n' > ~/.claude.json

# ---------------------------------------------------------------------------
# Run the agent. The prompt is the problem's PROMPT.md (which carries the
# solution.json contract); remove it afterward for a clean diff, exactly as
# swe-mini does. --max-turns is the enforceable iteration cap (can't run away
# like pi). --output-format stream-json --verbose emits one JSON event per
# line → agent.stream.jsonl, which trace_stats.py parses for
# tokens/turns/steps/compaction. `|| true` keeps the empty/failed-agent path
# working under `set -e` (measure.sh writes the FAIL facts later).
# ---------------------------------------------------------------------------
PROMPT=$(cat PROMPT.md)
rm PROMPT.md

tree -a

EFFORT_FLAG=()
[ -n "$EVAL_EFFORT" ] && EFFORT_FLAG=(--effort "$EVAL_EFFORT")

claude -p "$PROMPT" \
    --max-turns "$EVAL_MAX_TURNS" \
    "${EFFORT_FLAG[@]}" \
    --permission-mode bypassPermissions \
    --allowedTools "Read,Edit,Write,Bash" \
    --output-format stream-json --verbose \
    > "$PROBLEM_DIR/agent.stream.jsonl" 2> "$PROBLEM_DIR/agent.stderr.log" || true
