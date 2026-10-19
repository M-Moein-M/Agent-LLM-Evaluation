#!/bin/bash
# Endpoint test: does this endpoint work with all four agents, and what does it cost?
#
#   tests/endpoint/run.sh --endpoint NAME --model KEY [--agents claude,kimi,pi,swe-mini]
#                         [--steps 8] [--problem tiled-gemm] [--yes] [--l0-only]
#
# L0 sends protocol probes (tests/endpoint/probe.py). L1 runs one real round of a problem
# per agent with a step cap and checks the traces (tests/endpoint/check_trace.py). It
# runs on a login node: EVAL_GPU=0, so the measure phase fails by design and is ignored.
# The question is whether parsing, tool calling, thinking traces and token accounting
# work, not whether the agent solves the problem. Results: stdout and tests/endpoint/last-run.md.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ENDPOINT="${EVAL_ENDPOINT:-}"; MODEL=""; AGENTS="claude,kimi,pi,swe-mini"; STEPS=8
PROBLEM="tiled-gemm"; YES=0; L0_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --endpoint) ENDPOINT="$2"; shift 2 ;;
        --model)    MODEL="$2"; shift 2 ;;
        --agents)   AGENTS="$2"; shift 2 ;;
        --steps)    STEPS="$2"; shift 2 ;;
        --problem)  PROBLEM="$2"; shift 2 ;;
        --yes|-y)   YES=1; shift ;;
        --l0-only)  L0_ONLY=1; shift ;;
        -h|--help)  sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$MODEL" ]] || { echo "ERROR: --model KEY is required (see: python3 scripts/lib/resolve_endpoint.py --list)" >&2; exit 2; }
RESOLVE=(python3 scripts/lib/resolve_endpoint.py)
[[ -n "$ENDPOINT" ]] && export EVAL_ENDPOINT="$ENDPOINT"
"${RESOLVE[@]}" --models "$MODEL" --emit json > /dev/null || exit 2
IFS=',' read -r -a AGENT_LIST <<< "$AGENTS"

STAMP="endpoint-test-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$ROOT/eval-logs/endpoint-test/$STAMP"
mkdir -p "$RUN_DIR"

cat <<EOF
Endpoint test: endpoint=${ENDPOINT:-<only one configured>} model=$MODEL agents=$AGENTS
  L0: 9 short probes, about 1K input / 2.5K output tokens.
  L1: ${#AGENT_LIST[@]} agent(s), one $PROBLEM round each, capped at $STEPS steps/turns.
      Measured with a reasoning model, 8 steps: 7K-250K input and 0.6K-55K output per agent, about
      520K input and 100K output for all four. Worst case about 1M input and 250K output.
      (Input is summed over model calls. See tests/endpoint/README.md for a measured run.)
  Logs: $RUN_DIR
EOF
if [[ $YES -ne 1 ]]; then
    [[ -t 0 ]] || { echo "ERROR: not a terminal, pass --yes to confirm the token spend" >&2; exit 2; }
    read -r -p "Run it? [y/N] " ans; [[ "$ans" =~ ^[Yy] ]] || exit 0
fi

echo "== L0 =="
python3 tests/endpoint/probe.py --model "$MODEL" --json "$RUN_DIR/l0.json"
L0_RC=$?
if [[ $L0_RC -ne 0 ]]; then
    echo "L0 has FAILs. Fix those before L1 (the agents will fail the same way)." >&2
    python3 tests/endpoint/report.py --dir "$RUN_DIR" --meta "endpoint=${ENDPOINT:-default} model=$MODEL (L1 not run)" > /dev/null
    exit 1
fi

if [[ $L0_ONLY -ne 1 ]]; then
    for AGENT in "${AGENT_LIST[@]}"; do
        echo "== L1 $AGENT =="
        SIF="$ROOT/agents/apptainer/$AGENT/agent.sif"
        if [[ ! -f "$SIF" ]]; then
            echo "  SKIP: $SIF not built (apptainer build --fakeroot $SIF agents/apptainer/$AGENT/agent.def)"
            echo "{\"skipped\": \"agent.sif not built\"}" > "$RUN_DIR/$AGENT.json"
            continue
        fi
        (
            export EVAL_AGENT="$AGENT" CONTAINER_RUNTIME=apptainer EVAL_MODELS="$MODEL"
            export EVAL_MODEL_TAG="endpoint-test" EVAL_DATE_TAG="$STAMP" EVAL_ROUNDS=1 EVAL_SOLVE_PAR=1
            export EVAL_PROBLEMS="$PROBLEM" EVAL_GPU=0 EVAL_SKIP_PREFLIGHT=1
            case "$AGENT" in
                pi)       export EVAL_STEP_LIMIT="$STEPS" ;;
                claude|kimi) export EVAL_MAX_TURNS="$STEPS" ;;
                swe-mini)
                    # mini's step cap lives in mini.yaml; use a trimmed copy.
                    sed -E "s/^([[:space:]]*)step_limit:.*/\1step_limit: $STEPS/" \
                        agents/apptainer/swe-mini/mini.yaml > "$RUN_DIR/mini.yaml"
                    export MINI_CONFIG="$RUN_DIR/mini.yaml" ;;
            esac
            bash scripts/run_eval.sh > "$RUN_DIR/$AGENT.out" 2>&1
        )
        RC=$?
        echo "  run_eval.sh exit=$RC (a failed measure phase is expected without a GPU), log: $RUN_DIR/$AGENT.out"
        RD="$ROOT/traces/$PROBLEM/$AGENT/endpoint-test/$STAMP/round-1"
        REASONING=$("${RESOLVE[@]}" --models "$MODEL" --agent "$AGENT" --emit json | python3 -c 'import json,sys; print(int(json.load(sys.stdin)["models"][0]["reasoning"]))')
        python3 tests/endpoint/check_trace.py --round "$RD" --agent "$AGENT" $([[ $REASONING -eq 1 ]] && echo --reasoning) --json "$RUN_DIR/$AGENT.json" \
            || true
        [[ -f "$RUN_DIR/$AGENT.json" ]] || echo "{\"checks\": [{\"check\": \"trace\", \"status\": \"FAIL\", \"detail\": \"no round directory at $RD, see $RUN_DIR/$AGENT.out\"}], \"totals\": {}}" > "$RUN_DIR/$AGENT.json"
    done
fi

echo
python3 tests/endpoint/report.py --dir "$RUN_DIR" \
    --meta "endpoint=${ENDPOINT:-default} model=$MODEL problem=$PROBLEM steps=$STEPS date=$(date -u +%Y-%m-%d)"
