#! /bin/bash
# Phase 1 (SOLVE) — swe-mini agent. Mirror of agents/docker/pi/solve.sh; only
# the agent-invocation block differs. Runs mini-SWE-agent against our
# OpenAI-compatible gateway to produce code + solution.json + agent.traj.json.
#
# GPU-contention-tolerant: the agent's own compile/run iterations only need
# correctness feedback, so several solve containers can share the GPU. The
# graded measurement happens later, isolated, in measure.sh.

set -euo pipefail

PROBLEM_DIR=$1
EVAL_MODEL="${EVAL_MODEL:-unknown-model}"

cd "$PROBLEM_DIR"
rm -rf .git/
rm -rf .github/

# ---------------------------------------------------------------------------
# Run the agent (mini-SWE-agent / litellm against our OpenAI-compatible
# gateway). The same models.json pi uses is bind-mounted at /models.json and
# is the single source of the baseUrl + apiKey. mini just runs the bash the
# model emits, so it writes solution.json exactly as the prompt asks.
# ---------------------------------------------------------------------------
PROMPT=$(cat PROMPT.md)
rm PROMPT.md

tree -a

# Skip mini's first-run interactive setup wizard (no TTY in the container) —
# we supply the model + key + base URL via env and -c overrides below.
export MSWEA_CONFIGURED=true
export OPENAI_API_KEY="$(jq -r '.providers.local.apiKey' /models.json)"
API_BASE="$(jq -r '.providers.local.baseUrl' /models.json)"

# NOTE on max output tokens: we deliberately do NOT pass max_tokens here. pi's
# per-model maxTokens in models.json (e.g. 262144 for Qwen3.6-35B-A3B) is pi's
# internal "max output capability" and is *larger* than the model's
# contextWindow (256000). Forwarding it to litellm as max_completion_tokens is
# rejected by the server ("max_completion_tokens is too large: 262144. This
# model supports at most 256000") and kills every request. The server's own cap
# (= context window) is the correct ceiling, so we let it govern by omission.
#
# mini's default per-command timeout is 30s — far too short for CUDA builds —
# so bump environment.timeout. cwd is set to the problem dir so each (stateless)
# command starts there. cost_tracking=ignore_errors because our gateway models
# aren't in litellm's price registry. `|| true` keeps the empty/failed-agent
# path working under set -euo pipefail (measure.sh writes the FAIL facts later).
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
