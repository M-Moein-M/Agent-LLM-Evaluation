#! /bin/bash -l

# claude (Claude Code) agent orchestrator — APPTAINER runtime (cluster / SLURM).
# Two-phase mirror of agents/apptainer/swe-mini/host.sh (see the pi apptainer
# host.sh for the full rationale on the docker->apptainer deltas).
#
# claude specifics vs swe-mini: the agent is Claude Code, configured purely via
# env (set in this dir's solve.sh) so EVAL_MODELS works at submit time — no
# mini.yaml. It still reads the gateway baseUrl + apiKey from /models.json
# (bind-mounted) and gets a clean staging HOME so its ~/.claude state doesn't
# land in your real home. The in-container measure.sh is agent-agnostic and is
# reused verbatim from agents/docker/swe-mini/.
#
# IMPORTANT (vs swe-mini's `--home …:/root`): Claude Code's native installer puts
# the binary under /root/.local (bin/claude -> /root/.local/share/claude/...).
# Binding the staging home OVER /root would shadow it ("claude: command not
# found"). So we mount the staging home at its OWN path (HOME=$CLAUDEHOME) and
# leave the image's /root — and thus claude — intact. swe-mini's `mini` lives in
# /usr/local/bin, so it can map onto /root; claude cannot.

AGENT="claude"
ROOT="$(pwd)"
PROBLEM_DIR="$ROOT/problems"
TRACE_ROOT="$ROOT/traces"
AGENT_SIF="${AGENT_SIF:-$ROOT/agents/apptainer/claude/agent.sif}"
SOLVE_SCRIPT="$ROOT/agents/apptainer/claude/solve.sh"
MEASURE_SCRIPT="$ROOT/agents/docker/swe-mini/measure.sh"   # agent-agnostic, shared
MODELS_CONFIG="${MODELS_CONFIG:-$ROOT/agents/apptainer/pi/models.json}"
EVAL_ROUNDS="${EVAL_ROUNDS:-20}"
EVAL_PROBLEMS="${EVAL_PROBLEMS:-RabbitCT-LolaBunny}"
EVAL_SIZE="${EVAL_SIZE:-256}"
EVAL_VARIANT="${EVAL_VARIANT:-LolaBunnyGPU}"
EVAL_SOLVE_PAR="${EVAL_SOLVE_PAR:-4}"
EVAL_MAX_TURNS="${EVAL_MAX_TURNS:-120}"
# Selects providers.<name> in models.json (default "local" = gateway; "llama" =
# host). Forwarded via --env below — apptainer exec does NOT inherit host env.
EVAL_PROVIDER="${EVAL_PROVIDER:-local}"
# Claude Code token knobs (consumed by solve.sh). Empty = use solve.sh's own
# default (output cap 64000; thinking unset = max). MUST be forwarded explicitly
# via --env below — apptainer exec does NOT inherit the host environment here.
EVAL_MAX_OUTPUT_TOKENS="${EVAL_MAX_OUTPUT_TOKENS:-}"
EVAL_MAX_THINKING_TOKENS="${EVAL_MAX_THINKING_TOKENS:-}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"

DATE_TAG=$(date +%Y%m%d-%H%M%S)
echo "ROOT directory was set to $ROOT"
[[ -d "$PROBLEM_DIR" ]] || { echo "ERROR: Run from repo root. Missing $PROBLEM_DIR" >&2; exit 1; }

if [[ ! -f "$AGENT_SIF" ]]; then
    echo "ERROR: agent image not found: $AGENT_SIF" >&2
    echo "Build it first (you have fakeroot):" >&2
    echo "  APPTAINER_CACHEDIR=\$TMPDIR/apptainer-cache apptainer build --fakeroot \\" >&2
    echo "    $AGENT_SIF $ROOT/agents/apptainer/claude/agent.def" >&2
    exit 1
fi

WORK_ROOT=$(mktemp -d)
trap 'rm -rf "$WORK_ROOT"' EXIT

if [ -z "${EVAL_MODELS:-}" ]; then
    if command -v jq >/dev/null 2>&1; then
        EVAL_MODELS=$(jq -r '.providers[].models[].id' "$MODELS_CONFIG" | paste -sd, -)
    else
        echo "ERROR: EVAL_MODELS is not set and jq is unavailable to parse $MODELS_CONFIG" >&2
        exit 1
    fi
fi

IFS=',' read -r -a MODELS <<< "$EVAL_MODELS"
IFS=',' read -r -a PROBLEMS <<< "$EVAL_PROBLEMS"

job_workdir()  { echo "$WORK_ROOT/work/$1/round-$2/$3"; }
job_home()     { echo "$WORK_ROOT/home/$1/round-$2/$3"; }
job_tracedir() { echo "$TRACE_ROOT/$3/$AGENT/$1/$DATE_TAG/round-$2"; }

JOBS=()
for MODEL in "${MODELS[@]}"; do
    for ROUND in $(seq 1 "$EVAL_ROUNDS"); do
        for PROBLEM_NAME in "${PROBLEMS[@]}"; do
            if [[ ! -d "$PROBLEM_DIR/$PROBLEM_NAME" ]]; then
                echo "Skipping missing problem: $PROBLEM_DIR/$PROBLEM_NAME" >&2
                continue
            fi
            JOBS+=("$MODEL|$ROUND|$PROBLEM_NAME")
        done
    done
done

solve_job() {
    local MODEL="$1" ROUND="$2" PROBLEM_NAME="$3"
    local MODEL_TAG="${MODEL//\//_}"
    local PROBLEM="$PROBLEM_DIR/$PROBLEM_NAME"
    local WORK_DIR; WORK_DIR="$(job_workdir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
    local CLAUDEHOME; CLAUDEHOME="$(job_home "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"

    echo "[solve] problem: $PROBLEM_NAME | agent: $AGENT | model: $MODEL | round: $ROUND"
    mkdir -p "$WORK_DIR" "$CLAUDEHOME"
    cp -a "$PROBLEM"/. "$WORK_DIR"/

    local DATA_BIND=()
    if [[ -d "$DATA_ROOT/$PROBLEM_NAME/RabbitInput" ]]; then
        DATA_BIND=(--bind "$DATA_ROOT/$PROBLEM_NAME/RabbitInput:/problem/RabbitInput:ro")
    fi

    apptainer exec --nv \
        --home "$CLAUDEHOME" \
        --env "EVAL_AGENT=$AGENT" \
        --env "EVAL_MODEL=$MODEL" \
        --env "EVAL_ROUND=$ROUND" \
        --env "EVAL_SIZE=$EVAL_SIZE" \
        --env "EVAL_VARIANT=$EVAL_VARIANT" \
        --env "EVAL_MAX_TURNS=$EVAL_MAX_TURNS" \
        --env "EVAL_MAX_OUTPUT_TOKENS=$EVAL_MAX_OUTPUT_TOKENS" \
        --env "EVAL_MAX_THINKING_TOKENS=$EVAL_MAX_THINKING_TOKENS" \
        --env "EVAL_PROVIDER=$EVAL_PROVIDER" \
        --bind "$WORK_DIR:/problem" \
        --bind "$MODELS_CONFIG:/models.json:ro" \
        --bind "$SOLVE_SCRIPT:/solve.sh:ro" \
        "${DATA_BIND[@]}" \
        "$AGENT_SIF" /bin/bash -l /solve.sh /problem
}

measure_job() {
    local MODEL="$1" ROUND="$2" PROBLEM_NAME="$3"
    local MODEL_TAG="${MODEL//\//_}"
    local PROBLEM="$PROBLEM_DIR/$PROBLEM_NAME"
    local WORK_DIR;  WORK_DIR="$(job_workdir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
    local CLAUDEHOME; CLAUDEHOME="$(job_home "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
    local TRACE_DIR; TRACE_DIR="$(job_tracedir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"

    echo "[measure] problem: $PROBLEM_NAME | agent: $AGENT | model: $MODEL | round: $ROUND"

    local DATA_BIND=()
    if [[ -d "$DATA_ROOT/$PROBLEM_NAME/RabbitInput" ]]; then
        DATA_BIND=(--bind "$DATA_ROOT/$PROBLEM_NAME/RabbitInput:/problem/RabbitInput:ro")
    fi

    # Harness-owned perf bench (per problem). Bound read-only so the agent can
    # never edit what grades it; measure.sh runs the nsys/cuBLAS perf grading
    # only when this file is present.
    local GRADE_BIND=()
    if [[ -f "$ROOT/checks/$PROBLEM_NAME/bench.cu" ]]; then
        GRADE_BIND=(--bind "$ROOT/checks/$PROBLEM_NAME/bench.cu:/grade/bench.cu:ro")
    fi

    apptainer exec --nv \
        --env "EVAL_AGENT=$AGENT" \
        --env "EVAL_MODEL=$MODEL" \
        --env "EVAL_ROUND=$ROUND" \
        --env "EVAL_SIZE=$EVAL_SIZE" \
        --env "EVAL_VARIANT=$EVAL_VARIANT" \
        --bind "$WORK_DIR:/problem" \
        --bind "$MEASURE_SCRIPT:/measure.sh:ro" \
        "${DATA_BIND[@]}" \
        "${GRADE_BIND[@]}" \
        "$AGENT_SIF" /bin/bash -l /measure.sh /problem

    mkdir -p "$TRACE_DIR"
    rsync -a --prune-empty-dirs --compare-dest="$PROBLEM/" "$WORK_DIR/" "$TRACE_DIR/"
    diff -ru --exclude=".git" "$PROBLEM/" "$WORK_DIR/" > "$TRACE_DIR/changed_diff.txt" 2>/dev/null || true

    if [ ! -s "$TRACE_DIR/.eval/facts.json" ]; then
        echo "ERROR: no .eval/facts.json in trace for ${PROBLEM_NAME} | ${MODEL} | round-${ROUND} — extraction produced no evidence. Aborting eval." >&2
        exit 1
    fi
    cat "$TRACE_DIR/changed_diff.txt" 2>/dev/null || true

    rm -rf "$WORK_DIR" "$CLAUDEHOME"
}

echo "=== Phase 1: solving ${#JOBS[@]} job(s), up to $EVAL_SOLVE_PAR in parallel (shared GPU) ==="
running=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r MODEL ROUND PROBLEM_NAME <<< "$job"
    solve_job "$MODEL" "$ROUND" "$PROBLEM_NAME" &
    running=$((running + 1))
    if (( running >= EVAL_SOLVE_PAR )); then
        wait -n
        running=$((running - 1))
    fi
done
wait

echo "=== Phase 2: measuring ${#JOBS[@]} job(s), serial (GPU-isolated) ==="
for job in "${JOBS[@]}"; do
    IFS='|' read -r MODEL ROUND PROBLEM_NAME <<< "$job"
    measure_job "$MODEL" "$ROUND" "$PROBLEM_NAME"
done

if command -v python3 >/dev/null 2>&1; then
    python3 "$ROOT/checks/check_pass.py" --traces "$TRACE_ROOT"
    python3 "$ROOT/checks/trace_stats.py" --traces "$TRACE_ROOT"
else
    echo "WARN: python3 not found on host — skipping pass checks. Run checks/check_pass.py + trace_stats.py manually." >&2
fi
