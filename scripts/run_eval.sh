#!/bin/bash -l

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

export EVAL_MODELS
export EVAL_ROUNDS
export EVAL_PROBLEMS
export EVAL_SIZE
export EVAL_VARIANT
export MODELS_CONFIG

exec "$ROOT/agents/docker/pi/host.sh"
