#!/bin/bash
# Sourced by scripts/run_eval.sh and every agents/apptainer/<agent>/host.sh.
#
# resolve_endpoints <repo-root> <agent>
#   endpoints.yaml + .env  ->  $ROOT/.run/models.json (pi schema, read by the agents' solve.sh)
#   and exports EVAL_ENDPOINT, EVAL_PROVIDER, EVAL_MODELS, MODELS_CONFIG (+ EVAL_MODEL_TAG).
# Does nothing when MODELS_CONFIG is already set (an earlier caller resolved it, or you
# supply your own pi-schema models.json). claude-pro uses a subscription, not an endpoint.
resolve_endpoints() {
    local root="$1" agent="$2" out cfg
    [ "$agent" = "claude-pro" ] && return 0
    [ -n "${MODELS_CONFIG:-}" ] && return 0
    cfg="${EVAL_ENDPOINTS_FILE:-$root/endpoints.yaml}"
    if [ ! -f "$cfg" ]; then
        echo "ERROR: $cfg not found. Run: cp endpoints.yaml.example endpoints.yaml && cp .env.example .env" >&2
        echo "       (or set MODELS_CONFIG to your own pi-schema models.json)" >&2
        return 1
    fi
    out=$(python3 "$root/scripts/lib/resolve_endpoint.py" --agent "$agent" --emit sh \
            --out-models "$root/.run/models.json") || return 1
    eval "$out"
}
