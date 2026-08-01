#! /bin/bash -l

# swe-mini agent orchestrator — APPTAINER runtime (cluster / SLURM). Two-phase
# mirror of agents/docker/swe-mini/host.sh, adapted for Apptainer (see the pi
# apptainer host.sh for the full rationale on the docker->apptainer deltas).
#
# swe-mini specifics vs pi: no ~/.pi config — mini reads /models.json (jq) +
# /mini.yaml (both bind-mounted) and gets the key/base via env in solve.sh. We
# still hand it a clean staging HOME so litellm's cache doesn't land in your real
# home. The in-container solve.sh/measure.sh are reused from agents/docker/swe-mini/.

AGENT="swe-mini"
ROOT="$(pwd)"
PROBLEM_DIR="$ROOT/problems"
TRACE_ROOT="$ROOT/traces"
AGENT_SIF="${AGENT_SIF:-$ROOT/agents/apptainer/swe-mini/agent.sif}"
SOLVE_SCRIPT="$ROOT/agents/docker/swe-mini/solve.sh"
MEASURE_SCRIPT="$ROOT/agents/measure.sh"
MINI_CONFIG="${MINI_CONFIG:-$ROOT/agents/docker/swe-mini/mini.yaml}"
MODELS_CONFIG="${MODELS_CONFIG:-$ROOT/agents/apptainer/pi/models.json}"
EVAL_ROUNDS="${EVAL_ROUNDS:-20}"
EVAL_PROBLEMS="${EVAL_PROBLEMS:-RabbitCT-LolaBunny}"
EVAL_SIZE="${EVAL_SIZE:-256}"
EVAL_VARIANT="${EVAL_VARIANT:-LolaBunnyGPU}"
EVAL_SOLVE_PAR="${EVAL_SOLVE_PAR:-4}"
# Selects providers.<name> in models.json (default "local" = gateway; "llama" =
# host). Forwarded via --env below. NB: apptainer exec inherits the host environment by DEFAULT; we pass --cleanenv
#   to block that, so anything needed inside must be forwarded explicitly.
EVAL_PROVIDER="${EVAL_PROVIDER:-local}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"

DATE_TAG=$(date +%Y%m%d-%H%M%S)
echo "ROOT directory was set to $ROOT"
[[ -d "$PROBLEM_DIR" ]] || { echo "ERROR: Run from repo root. Missing $PROBLEM_DIR" >&2; exit 1; }

if [[ ! -f "$AGENT_SIF" ]]; then
    echo "ERROR: agent image not found: $AGENT_SIF" >&2
    echo "Build it first (you have fakeroot):" >&2
    echo "  APPTAINER_CACHEDIR=\$TMPDIR/apptainer-cache apptainer build --fakeroot \\" >&2
    echo "    $AGENT_SIF $ROOT/agents/apptainer/swe-mini/agent.def" >&2
    exit 1
fi

# GPU passthrough is conditional. The harness now also runs CPU-only problems
# (e.g. 2D-seq on the test cluster), where --nv on a node without an NVIDIA
# driver is at best noise. auto = pass --nv only when a GPU is actually visible;
# EVAL_GPU=1 forces it on, EVAL_GPU=0 forces it off.
NV_FLAG=()
if [[ "${EVAL_GPU:-auto}" == "1" ]] || { [[ "${EVAL_GPU:-auto}" == "auto" ]] &&
      command -v nvidia-smi >/dev/null 2>&1 &&
      nvidia-smi -L 2>/dev/null | grep -q "^GPU"; }; then
    NV_FLAG=(--nv)
fi
echo "GPU passthrough: ${NV_FLAG[*]:-none (CPU-only run)}"

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
    local SWEHOME;  SWEHOME="$(job_home "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"

    echo "[solve] problem: $PROBLEM_NAME | agent: $AGENT | model: $MODEL | round: $ROUND"
    mkdir -p "$WORK_DIR" "$SWEHOME"
    # rsync, not `cp -a`: cp -a tries to replicate ACLs, and on the test-cluster
    # filesystem that fails with "preserving permissions ... Operation not
    # supported" for every single file — ~25 lines of noise per round, drowning
    # the log. rsync -a preserves what actually matters here (mode, mtime,
    # symlinks) and does not touch ACLs unless asked with -A. mtime matters: the
    # `rsync --compare-dest` in measure_job decides "did the agent modify this
    # file?" from size+mtime, so a copy that reset mtimes would make every file
    # look tampered with.
    rsync -a "$PROBLEM"/ "$WORK_DIR"/

    local DATA_BIND=()
    if [[ -d "$DATA_ROOT/$PROBLEM_NAME/RabbitInput" ]]; then
        DATA_BIND=(--bind "$DATA_ROOT/$PROBLEM_NAME/RabbitInput:/problem/RabbitInput:ro")
    fi

    apptainer exec --cleanenv "${NV_FLAG[@]}" \
        --home "$SWEHOME:/root" \
        --env "EVAL_AGENT=$AGENT" \
        --env "EVAL_MODEL=$MODEL" \
        --env "EVAL_ROUND=$ROUND" \
        --env "EVAL_SIZE=$EVAL_SIZE" \
        --env "EVAL_VARIANT=$EVAL_VARIANT" \
        --env "EVAL_PROVIDER=$EVAL_PROVIDER" \
        --env "EVAL_MODEL_CLASS=${EVAL_MODEL_CLASS:-}" \
        --bind "$WORK_DIR:/problem" \
        --bind "$MODELS_CONFIG:/models.json:ro" \
        --bind "$MINI_CONFIG:/mini.yaml:ro" \
        --bind "$SOLVE_SCRIPT:/solve.sh:ro" \
        "${DATA_BIND[@]}" \
        "$AGENT_SIF" /bin/bash -l /solve.sh /problem
}

measure_job() {
    local MODEL="$1" ROUND="$2" PROBLEM_NAME="$3"
    local MODEL_TAG="${MODEL//\//_}"
    local PROBLEM="$PROBLEM_DIR/$PROBLEM_NAME"
    local WORK_DIR;  WORK_DIR="$(job_workdir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
    local SWEHOME;   SWEHOME="$(job_home "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
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

    apptainer exec --cleanenv "${NV_FLAG[@]}" \
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

    rm -rf "$WORK_DIR" "$SWEHOME"
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
