#! /bin/bash
# Phase 1 (SOLVE) — swe-mini agent. Mirror of agents/apptainer/pi/solve.sh; only
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
# EVAL_PROVIDER names the providers.<name> block in /models.json (generated from
# endpoints.yaml by scripts/lib/resolve_endpoint.py) that supplies baseUrl + apiKey.
EVAL_PROVIDER="${EVAL_PROVIDER:?EVAL_PROVIDER is not set (run via scripts/run_eval.sh)}"
export OPENAI_API_KEY="$(jq -r ".providers.${EVAL_PROVIDER}.apiKey" /models.json)"
API_BASE="$(jq -r ".providers.${EVAL_PROVIDER}.baseUrl" /models.json)"

# NOTE on max output tokens: we deliberately do NOT pass max_tokens here. pi's
# per-model maxTokens in models.json can be larger than the model's contextWindow.
# Forwarding it to litellm as max_completion_tokens is rejected by servers that
# validate it ("max_completion_tokens is too large") and kills every request. The
# server's own cap (= context window) is the correct ceiling, so we let it govern
# by omission.
#
# EVAL_TOOL_CHOICE_BASH=1 pins tool_choice to mini's only tool (`bash`). Some
# models (seen with Llama-3.3-70B on SGLang) need it, or mini's function-calling
# agent dies:
#   1. Left to choose, the model writes the call as a fenced ```json block inside
#      `content` whenever it reasons first; the server's parser (correctly) ignores
#      that, so mini sees NO tool_calls and aborts (RepeatedFormatError).
#   2. With a plain tool_choice=required it emits MULTIPLE tool_calls in one turn;
#      the server then rejects the follow-up ("only supports single tool-calls at
#      once") and mini crashes.
# Pinning tool_choice to the named `bash` function forces exactly ONE structured
# call per turn. Set via dotted `-c` leaves, which deep-merge into model_kwargs.
# Off by default: the measured runs ran with tool_choice=auto (the earlier
# `EVAL_PROVIDER=llama` match never fired for the provider names actually used).
EXTRA_OPTS=()
if [ "${EVAL_TOOL_CHOICE_BASH:-0}" = "1" ]; then
	EXTRA_OPTS+=(-c "model.model_kwargs.tool_choice.type=function")
	EXTRA_OPTS+=(-c "model.model_kwargs.tool_choice.function.name=bash")
fi

# Optional non-default action mode. When EVAL_MODEL_CLASS is set (e.g.
# "litellm_textbased"), select that mini model class so actions are parsed from a
# ```mswea_bash_command``` block in the model's TEXT output instead of structured
# JSON tool-calls. Needed for models whose tool-call argument serialization is
# broken on the gateway (gpt-oss-120b emits command strings that aren't valid JSON
# -> RepeatedFormatError). The matching prompt config must be bound at /mini.yaml
# (host.sh: MINI_CONFIG=.../mini_textbased.yaml). Unset -> default toolcall path.
if [ -n "${EVAL_MODEL_CLASS:-}" ]; then
	EXTRA_OPTS+=(-c "model.model_class=$EVAL_MODEL_CLASS")
fi

# Optional step cap override (mini.yaml's default is 120). Used by smoke tests
# to bound the interaction to a handful of turns without needing a full solve.
if [ -n "${EVAL_STEP_LIMIT:-}" ]; then
	EXTRA_OPTS+=(-c "agent.step_limit=$EVAL_STEP_LIMIT")
fi

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
	"${EXTRA_OPTS[@]}" \
	-t "$PROMPT" \
	-o ~/agent.traj.json || true
cp ~/agent.traj.json "$PROBLEM_DIR" 2>/dev/null || true
