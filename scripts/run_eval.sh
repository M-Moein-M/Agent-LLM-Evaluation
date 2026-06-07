#!/bin/bash -l

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EVAL_AGENT="${EVAL_AGENT:-pi}"
EVAL_MODELS="${EVAL_MODELS:-}"
EVAL_ROUNDS="${EVAL_ROUNDS:-20}"
EVAL_PROBLEMS="${EVAL_PROBLEMS:-RabbitCT}"
EVAL_SIZE="${EVAL_SIZE:-256}"
EVAL_VARIANT="${EVAL_VARIANT:-LolaBunnyGPU}"
MODELS_CONFIG="${MODELS_CONFIG:-$ROOT/agents/apptainer/pi/models.json}"

if [ -z "$EVAL_MODELS" ]; then
  if command -v jq >/dev/null 2>&1; then
    EVAL_MODELS=$(jq -r '.providers[].models[].id' "$MODELS_CONFIG" | paste -sd, -)
  else
    echo "ERROR: EVAL_MODELS not set and jq not available to parse $MODELS_CONFIG" >&2
    exit 1
  fi
fi

export EVAL_AGENT
export EVAL_MODELS
export EVAL_ROUNDS
export EVAL_PROBLEMS
export EVAL_SIZE
export EVAL_VARIANT
export MODELS_CONFIG

DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export DATA_ROOT

# ---------------------------------------------------------------------------
# Stage 1: ensure each problem's bulk input data is present (hardcoded per
# problem for now). Data lives in the gitignored repo-root data/ dir and is
# bind-mounted read-only into the container by host.sh — never copied per round.
# ---------------------------------------------------------------------------
if [[ ",$EVAL_PROBLEMS," == *",RabbitCT,"* ]]; then
  RABBIT_DATA="$DATA_ROOT/RabbitCT"
  sz=$(du -sm "$RABBIT_DATA" 2>/dev/null | cut -f1 || true); sz="${sz:-0}"
  if [[ "$sz" -ge 4096 ]]; then
    echo "Stage 1: RabbitCT data present (${sz} MB in $RABBIT_DATA) — skipping download."
  else
    echo "Stage 1: RabbitCT data missing/incomplete (${sz} MB) — downloading into $RABBIT_DATA ..."
    "$ROOT/problems/RabbitCT/download-input.sh" "$RABBIT_DATA"
  fi
fi

# ---------------------------------------------------------------------------
# Stage 2: run the eval with the selected agent (EVAL_AGENT, default "pi").
# Each agent has its own orchestrator at agents/docker/<agent>/host.sh.
# ---------------------------------------------------------------------------
AGENT_HOST="$ROOT/agents/docker/$EVAL_AGENT/host.sh"
if [[ ! -f "$AGENT_HOST" ]]; then
  echo "ERROR: unknown EVAL_AGENT '$EVAL_AGENT' — no orchestrator at $AGENT_HOST" >&2
  exit 1
fi
exec "$AGENT_HOST"
