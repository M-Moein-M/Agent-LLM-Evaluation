#!/bin/bash -l

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# host.sh scripts treat the current directory as the repo root.
cd "$ROOT"

EVAL_AGENT="${EVAL_AGENT:-swe-mini}"               # claude | kimi | pi | swe-mini | claude-pro
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-apptainer}" # apptainer is the only supported runtime
EVAL_ROUNDS="${EVAL_ROUNDS:-20}"
EVAL_PROBLEMS="${EVAL_PROBLEMS:-RabbitCT-LolaBunny}"
EVAL_SIZE="${EVAL_SIZE:-256}"
EVAL_VARIANT="${EVAL_VARIANT:-LolaBunnyGPU}"

# Endpoint: endpoints.yaml + .env -> .run/models.json and EVAL_ENDPOINT / EVAL_PROVIDER /
# EVAL_MODELS / MODELS_CONFIG (see scripts/lib/resolve_endpoint.py). Select with
# EVAL_ENDPOINT=<name> and EVAL_MODELS=<key|id>[,...]; EVAL_MODELS defaults to every
# model of the endpoint. claude-pro takes a subscription model id instead.
. "$ROOT/scripts/lib/endpoint.sh"
if [ -z "${EVAL_MODELS:-}" ] && [ "$EVAL_AGENT" = "claude-pro" ]; then
  EVAL_MODELS="claude-opus-5"   # real Anthropic id, not an endpoint model
fi
export EVAL_AGENT CONTAINER_RUNTIME EVAL_ROUNDS EVAL_PROBLEMS EVAL_SIZE EVAL_VARIANT
resolve_endpoints "$ROOT" "$EVAL_AGENT" || exit 1
export EVAL_MODELS

DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export DATA_ROOT

# ---------------------------------------------------------------------------
# Stage 0: reachability preflight. Sends a 1-token request per model with the
# protocol the agent speaks (/chat/completions; /v1/messages for claude) and aborts
# in ~30 s if any model is unreachable, instead of discovering it hours into a batch.
# Skip with EVAL_SKIP_PREFLIGHT=1.
# ---------------------------------------------------------------------------
if [ "${EVAL_SKIP_PREFLIGHT:-0}" != "1" ] && [ "$EVAL_AGENT" = "claude-pro" ]; then
  # claude-pro runs Claude Code on a personal subscription against api.anthropic.com.
  # Check the token file and outbound HTTPS. Auth itself is validated by the run.
  TOKEN_FILE="${OAUTH_TOKEN_FILE:-$ROOT/agents/apptainer/claude-pro/.secrets/oauth_token}"
  echo "Stage 0: claude-pro preflight (subscription auth, api.anthropic.com)"
  if [ ! -s "$TOKEN_FILE" ]; then
    echo "ERROR: subscription token missing/empty: $TOKEN_FILE" >&2
    echo "       See agents/apptainer/claude-pro/README.md (claude setup-token)." >&2
    exit 1
  fi
  echo "  [ok]   token file present: $TOKEN_FILE"
  http=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 10 --max-time 30 \
    https://api.anthropic.com/v1/messages 2>/dev/null || echo 000)
  if [ "$http" = "000" ]; then
    echo "ERROR: cannot reach https://api.anthropic.com from this node (no route/proxy)." >&2
    exit 1
  fi
  echo "  [ok]   api.anthropic.com reachable (HTTP $http)"
elif [ "${EVAL_SKIP_PREFLIGHT:-0}" != "1" ] && [ -f "$ROOT/.run/models.json" ] && [ "$MODELS_CONFIG" = "$ROOT/.run/models.json" ]; then
  python3 "$ROOT/scripts/lib/resolve_endpoint.py" --agent "$EVAL_AGENT" --preflight || exit 1
fi

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
  # ln -sfn + || true, not a bare `ln -s` under the `! -e` guard: several
  # campaigns can now share one node (a packed SLURM job can run 3
  # concurrent run_eval.sh), so the check-then-create is a TOCTOU race — all
  # three see "missing", two lose the ln and die on `set -e`. -sfn makes it
  # idempotent, and the guard stays to keep the log quiet in the common case.
  for _v in "${RABBIT_VARIANTS[@]}"; do
    if [[ "$_v" != "RabbitCT" && ! -e "$DATA_ROOT/$_v" ]]; then
      ln -sfn RabbitCT "$DATA_ROOT/$_v" 2>/dev/null || true
      echo "Stage 1: linked data/$_v -> RabbitCT"
    fi
  done
fi

# ---------------------------------------------------------------------------
# Stage 2: run the eval with the selected agent + runtime. Orchestrator lives at
# agents/<CONTAINER_RUNTIME>/<EVAL_AGENT>/host.sh. It runs the two-phase contract
# (parallel solve / serial measure) over the problems and checks.
# ---------------------------------------------------------------------------
AGENT_HOST="$ROOT/agents/$CONTAINER_RUNTIME/$EVAL_AGENT/host.sh"
if [[ ! -f "$AGENT_HOST" ]]; then
  echo "ERROR: no orchestrator at $AGENT_HOST (EVAL_AGENT='$EVAL_AGENT', CONTAINER_RUNTIME='$CONTAINER_RUNTIME')" >&2
  exit 1
fi
exec "$AGENT_HOST"
