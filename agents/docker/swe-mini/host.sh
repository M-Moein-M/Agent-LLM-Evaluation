#! /bin/bash -l

# swe-mini agent orchestrator. Two-phase mirror of agents/docker/pi/host.sh;
# differs only in the agent-specific bits (image, solve script, config mounts,
# AGENT tag). Keep the shared logic (job list, parallel solve / serial measure,
# trace extraction, check_pass) in sync with the pi version.

AGENT="swe-mini"
ROOT="$(pwd)"
PROBLEM_DIR="$ROOT/problems"
TRACE_ROOT="$ROOT/traces"
DOCKERFILE="$ROOT/agents/docker/swe-mini/Dockerfile"
IMAGE="agent-eval-swe-mini:latest"
SOLVE_SCRIPT="$ROOT/agents/docker/swe-mini/solve.sh"
MEASURE_SCRIPT="$ROOT/agents/docker/swe-mini/measure.sh"
MINI_CONFIG="$ROOT/agents/docker/swe-mini/mini.yaml"
MODELS_CONFIG="${MODELS_CONFIG:-$ROOT/agents/apptainer/pi/models.json}"
EVAL_ROUNDS="${EVAL_ROUNDS:-20}"
EVAL_PROBLEMS="${EVAL_PROBLEMS:-RabbitCT-LolaBunny}"
EVAL_SIZE="${EVAL_SIZE:-256}"
EVAL_VARIANT="${EVAL_VARIANT:-LolaBunnyGPU}"
# Max concurrent agent-solve containers in Phase 1. They share the GPU: the
# agent's own compile/run iterations are correctness feedback, which tolerates
# contention, and most solve wall-clock is spent waiting on the LLM gateway
# (GPU idle). Phase 2 (the graded run) is ALWAYS serial + GPU-isolated, so this
# never affects the measured numbers. Set to 1 for the old fully-serial behavior.
EVAL_SOLVE_PAR="${EVAL_SOLVE_PAR:-4}"
# Large per-problem input data lives outside the (copied) problem dir, in a
# gitignored repo-root data/ dir, and is bind-mounted read-only into the
# container — never copied per round. Overridable via DATA_ROOT.
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"

DATE_TAG=$(date +%Y%m%d-%H%M%S)
echo "ROOT directory was set to $ROOT"
[[ -d "$PROBLEM_DIR" ]] || { echo "ERROR: Run from repo root. Missing $PROBLEM_DIR" >&2; exit 1; }
[[ -f "$DOCKERFILE" ]] || { echo "ERROR: Run from repo root. Missing $DOCKERFILE" >&2; exit 1; }

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -t "$IMAGE" -f "$DOCKERFILE" "$(dirname "$DOCKERFILE")"
fi

WORK_ROOT=$(mktemp -d)
trap 'rm -rf "$WORK_ROOT"' EXIT

if [ -z "$EVAL_MODELS" ]; then
    if command -v jq >/dev/null 2>&1; then
        EVAL_MODELS=$(jq -r '.providers[].models[].id' "$MODELS_CONFIG" | paste -sd, -)
    else
        echo "ERROR: EVAL_MODELS is not set and jq is unavailable to parse $MODELS_CONFIG" >&2
        exit 1
    fi
fi

IFS=',' read -r -a MODELS <<< "$EVAL_MODELS"
IFS=',' read -r -a PROBLEMS <<< "$EVAL_PROBLEMS"

# ---------------------------------------------------------------------------
# Per-job paths are derived deterministically from the (model, round, problem)
# tuple so Phase 1 and Phase 2 agree without sharing state. Each job gets its
# OWN workdir (keyed by model/round/problem) so parallel solves never clobber.
# ---------------------------------------------------------------------------
job_workdir()  { echo "$WORK_ROOT/$1/round-$2/$3"; }                       # MODEL_TAG ROUND PROBLEM_NAME
job_tracedir() { echo "$TRACE_ROOT/$3/$AGENT/$1/$DATE_TAG/round-$2"; }      # MODEL_TAG ROUND PROBLEM_NAME

# Build the job list once. Missing problems are skipped here so neither phase
# has to re-check.
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

# ---------------------------------------------------------------------------
# Phase 1 (SOLVE): run the mini agent. Containers share --gpus all and run up
# to EVAL_SOLVE_PAR at a time. Output (solution.json + sources + agent.traj.json)
# is left in each job's persistent workdir for Phase 2 to consume.
# ---------------------------------------------------------------------------
solve_job() {
    local MODEL="$1" ROUND="$2" PROBLEM_NAME="$3"
    local MODEL_TAG="${MODEL//\//_}"
    local PROBLEM="$PROBLEM_DIR/$PROBLEM_NAME"
    local WORK_DIR; WORK_DIR="$(job_workdir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"

    echo "[solve] problem: $PROBLEM_NAME | agent: $AGENT | model: $MODEL | round: $ROUND"
    mkdir -p "$WORK_DIR"
    cp -a "$PROBLEM"/. "$WORK_DIR"/

    # Read-only mount for problems whose bulk input data lives in $DATA_ROOT.
    local DATA_MOUNT=()
    if [[ -d "$DATA_ROOT/$PROBLEM_NAME/RabbitInput" ]]; then
        DATA_MOUNT=(-v "$DATA_ROOT/$PROBLEM_NAME/RabbitInput:/problem/RabbitInput:ro")
    fi

    docker run --rm \
        --gpus all \
        -e "EVAL_AGENT=$AGENT" \
        -e "EVAL_MODEL=$MODEL" \
        -e "EVAL_ROUND=$ROUND" \
        -e "EVAL_SIZE=$EVAL_SIZE" \
        -e "EVAL_VARIANT=$EVAL_VARIANT" \
        -v "$WORK_DIR:/problem" \
        -v "$MODELS_CONFIG:/models.json:ro" \
        -v "$MINI_CONFIG:/mini.yaml:ro" \
        -v "$SOLVE_SCRIPT:/solve.sh:ro" \
        "${DATA_MOUNT[@]}" \
        "$IMAGE" /bin/bash -l /solve.sh /problem
}

# ---------------------------------------------------------------------------
# Phase 2 (MEASURE): replay build/run with GPU sampling, then extract the
# trace. Runs SERIALLY so each measured run owns the GPU.
# ---------------------------------------------------------------------------
measure_job() {
    local MODEL="$1" ROUND="$2" PROBLEM_NAME="$3"
    local MODEL_TAG="${MODEL//\//_}"
    local PROBLEM="$PROBLEM_DIR/$PROBLEM_NAME"
    local WORK_DIR;  WORK_DIR="$(job_workdir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
    local TRACE_DIR; TRACE_DIR="$(job_tracedir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"

    echo "[measure] problem: $PROBLEM_NAME | agent: $AGENT | model: $MODEL | round: $ROUND"

    local DATA_MOUNT=()
    if [[ -d "$DATA_ROOT/$PROBLEM_NAME/RabbitInput" ]]; then
        DATA_MOUNT=(-v "$DATA_ROOT/$PROBLEM_NAME/RabbitInput:/problem/RabbitInput:ro")
    fi

    docker run --rm \
        --gpus all \
        -e "EVAL_AGENT=$AGENT" \
        -e "EVAL_MODEL=$MODEL" \
        -e "EVAL_ROUND=$ROUND" \
        -e "EVAL_SIZE=$EVAL_SIZE" \
        -e "EVAL_VARIANT=$EVAL_VARIANT" \
        -v "$WORK_DIR:/problem" \
        -v "$MEASURE_SCRIPT:/measure.sh:ro" \
        "${DATA_MOUNT[@]}" \
        "$IMAGE" /bin/bash -l /measure.sh /problem

    # ----------------------------------------------------------------
    # Extract the trace from INSIDE a container, streamed out as a tar
    # over stdout. Why not host-side rsync/diff: the eval container ran
    # as root, and on WSL2 the host's view of the bind mount lags the
    # container's writes — a path stat of .eval/facts.json can succeed
    # while a readdir of the workdir still omits that (newest) entry, and
    # rsync/diff discover files via readdir, so they intermittently
    # produced .eval-less, diff-only traces. A container's view of the
    # just-written workdir IS coherent, so we do the rsync/diff there and
    # pipe the result out as a tar; the host only WRITES TRACE_DIR from
    # that byte stream, so its own readdir is always coherent. The same
    # container also chowns the workdir back so the EXIT-trap cleanup (as
    # this user) can delete it. --entrypoint bash skips the CUDA image
    # banner so stdout is pure tar.
    # ----------------------------------------------------------------
    mkdir -p "$TRACE_DIR"
    local TRACE_TAR="$WORK_ROOT/trace-${MODEL_TAG}-${PROBLEM_NAME}-${ROUND}.tar"
    docker run --rm --entrypoint bash \
        -e "HOST_UID=$(id -u)" \
        -e "HOST_GID=$(id -g)" \
        -v "$WORK_DIR:/work" \
        -v "$PROBLEM:/orig:ro" \
        "$IMAGE" -c '
            set -e
            mkdir -p /stage
            rsync -a --prune-empty-dirs --compare-dest=/orig/ --exclude=".pi/agent/" /work/ /stage/
            diff -ru --exclude=".git" /orig/ /work/ > /stage/changed_diff.txt 2>/dev/null || true
            chown -R "$HOST_UID:$HOST_GID" /work
            tar -C /stage -cf - .
        ' > "$TRACE_TAR"

    tar -C "$TRACE_DIR" -xf "$TRACE_TAR"
    rm -f "$TRACE_TAR"

    # Fail LOUD if the agent produced no judgable evidence. measure.sh always
    # writes a .eval/facts.json sentinel (even on agent failure), so its
    # absence means extraction genuinely failed. Reads what we just wrote on
    # the host (coherent), not the bind mount.
    if [ ! -s "$TRACE_DIR/.eval/facts.json" ]; then
        echo "ERROR: no .eval/facts.json in trace for ${PROBLEM_NAME} | ${MODEL} | round-${ROUND} — extraction produced no evidence. Aborting eval." >&2
        exit 1
    fi
    cat "$TRACE_DIR/changed_diff.txt" 2>/dev/null || true

    # Trace is safely on the host now — free this job's workdir.
    rm -rf "$WORK_DIR"
}

# ---------------------------------------------------------------------------
# Phase 1: solve, up to EVAL_SOLVE_PAR containers at once. A simple wait -n
# slot pool caps in-flight jobs without a job-control framework.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Phase 2: measure, strictly serial so every graded run owns the GPU.
# ---------------------------------------------------------------------------
echo "=== Phase 2: measuring ${#JOBS[@]} job(s), serial (GPU-isolated) ==="
for job in "${JOBS[@]}"; do
    IFS='|' read -r MODEL ROUND PROBLEM_NAME <<< "$job"
    measure_job "$MODEL" "$ROUND" "$PROBLEM_NAME"
done

# ---------------------------------------------------------------------------
# Host-side pass judgment over the captured evidence (no GPU, re-runnable).
# Re-run any time after tuning a threshold in checks/<PROBLEM>/check.py:
#     python "$ROOT/checks/check_pass.py" --traces "$TRACE_ROOT"
# ---------------------------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    python3 "$ROOT/checks/check_pass.py" --traces "$TRACE_ROOT"
else
    echo "WARN: python3 not found on host — skipping pass checks. Run checks/check_pass.py manually." >&2
fi
