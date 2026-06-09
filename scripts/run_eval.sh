#!/bin/bash -l

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EVAL_AGENT="${EVAL_AGENT:-pi}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"   # docker (local) | apptainer (cluster)
EVAL_MODELS="${EVAL_MODELS:-}"
EVAL_ROUNDS="${EVAL_ROUNDS:-20}"
EVAL_PROBLEMS="${EVAL_PROBLEMS:-RabbitCT-LolaBunny}"
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
export CONTAINER_RUNTIME
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
# RabbitCT family: every RabbitCT-<variant> problem reconstructs from the SAME
# ~4.5GB RabbitInput. Keep one canonical copy in data/RabbitCT/ and symlink each
# requested variant's data dir (data/<variant> -> RabbitCT) so host.sh's
# per-problem lookup ($DATA_ROOT/<problem>/RabbitInput) resolves to it — one
# download, no duplication.
RABBIT_CANON="$DATA_ROOT/RabbitCT"
RABBIT_VARIANTS=()
IFS=',' read -r -a _PROBS <<< "$EVAL_PROBLEMS"
for _p in "${_PROBS[@]}"; do
  case "$_p" in
    RabbitCT|RabbitCT-*) RABBIT_VARIANTS+=("$_p") ;;
  esac
done
if [[ ${#RABBIT_VARIANTS[@]} -gt 0 ]]; then
  sz=$(du -sLm "$RABBIT_CANON" 2>/dev/null | cut -f1 || true); sz="${sz:-0}"
  if [[ "$sz" -ge 4096 ]]; then
    echo "Stage 1: RabbitCT data present (${sz} MB in $RABBIT_CANON) — skipping download."
  else
    echo "Stage 1: RabbitCT data missing/incomplete (${sz} MB) — downloading into $RABBIT_CANON ..."
    mkdir -p "$RABBIT_CANON"
    "$ROOT/problems/RabbitCT-LolaBunny/download-input.sh" "$RABBIT_CANON"
  fi
  # Point each requested variant's data dir at the canonical copy (gitignored).
  for _v in "${RABBIT_VARIANTS[@]}"; do
    if [[ "$_v" != "RabbitCT" && ! -e "$DATA_ROOT/$_v" ]]; then
      ln -s RabbitCT "$DATA_ROOT/$_v"
      echo "Stage 1: linked data/$_v -> RabbitCT"
    fi
  done
fi

# ---------------------------------------------------------------------------
# Stage 2: run the eval with the selected agent + runtime. Orchestrator lives at
# agents/<CONTAINER_RUNTIME>/<EVAL_AGENT>/host.sh — docker (local) or apptainer
# (cluster). Both implement the same two-phase (parallel solve / serial measure)
# contract over the same problems + checks.
# ---------------------------------------------------------------------------
AGENT_HOST="$ROOT/agents/$CONTAINER_RUNTIME/$EVAL_AGENT/host.sh"
if [[ ! -f "$AGENT_HOST" ]]; then
  echo "ERROR: no orchestrator at $AGENT_HOST (EVAL_AGENT='$EVAL_AGENT', CONTAINER_RUNTIME='$CONTAINER_RUNTIME')" >&2
  exit 1
fi
exec "$AGENT_HOST"
