#! /bin/bash -l

# pi agent orchestrator — APPTAINER runtime (cluster / SLURM). Two-phase mirror
# of agents/docker/pi/host.sh, adapted for Apptainer:
#   * docker run --gpus all   ->  apptainer exec --nv
#   * -e K=V / -v S:D[:ro]     ->  --env K=V / --bind S:D[:ro]
#   * image build/inspect      ->  prebuilt .sif (built separately with fakeroot)
#   * pi config (~/.pi/agent)  ->  per-job --home staging dir (Apptainer runs as
#                                  YOU, HOME defaults to your real home, so we
#                                  point HOME at a staging dir holding models.json)
#   * trace extraction         ->  plain host-side rsync/diff. The Docker version
#                                  did this inside a root container to dodge a
#                                  WSL2 bind-mount readdir race + root-owned files;
#                                  on a real Linux node Apptainer writes files as
#                                  YOU and the FS view is coherent, so none of that
#                                  is needed.
#
# The in-container scripts (solve.sh, measure.sh) are runtime-agnostic and reused
# verbatim from agents/docker/pi/ — keep them as the single source of truth.

AGENT="pi"
ROOT="$(pwd)"
PROBLEM_DIR="$ROOT/problems"
TRACE_ROOT="$ROOT/traces"
AGENT_SIF="${AGENT_SIF:-$ROOT/agents/apptainer/pi/agent.sif}"
# Shared, runtime-agnostic in-container scripts (canonical copy under docker/).
SOLVE_SCRIPT="$ROOT/agents/docker/pi/solve.sh"
MEASURE_SCRIPT="$ROOT/agents/measure.sh"
MODELS_CONFIG="${MODELS_CONFIG:-$ROOT/agents/apptainer/pi/models.json}"
EVAL_ROUNDS="${EVAL_ROUNDS:-20}"
EVAL_PROBLEMS="${EVAL_PROBLEMS:-RabbitCT-LolaBunny}"
EVAL_SIZE="${EVAL_SIZE:-256}"
EVAL_VARIANT="${EVAL_VARIANT:-LolaBunnyGPU}"
# Max concurrent agent-solve processes in Phase 1. They share the GPU; the
# agent's own compile/run iterations only need correctness feedback (which
# tolerates contention) and most solve time is spent waiting on the LLM gateway.
# Phase 2 (the graded run) is ALWAYS serial + GPU-isolated. On an A40/A100 the
# real ceiling is node CPU/RAM + gateway throughput, not GPU memory — so this can
# go high (32+). Set to 1 for fully-serial.
EVAL_SOLVE_PAR="${EVAL_SOLVE_PAR:-4}"
# Hard cap on assistant turns per solve (pi fork's --step-limit). Forwarded into
# the container and consumed by solve.sh.
EVAL_STEP_LIMIT="${EVAL_STEP_LIMIT:-120}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"

DATE_TAG=$(date +%Y%m%d-%H%M%S)
echo "ROOT directory was set to $ROOT"
[[ -d "$PROBLEM_DIR" ]] || { echo "ERROR: Run from repo root. Missing $PROBLEM_DIR" >&2; exit 1; }

if [[ ! -f "$AGENT_SIF" ]]; then
    echo "ERROR: agent image not found: $AGENT_SIF" >&2
    echo "Build it first (you have fakeroot):" >&2
    echo "  APPTAINER_CACHEDIR=\$TMPDIR/apptainer-cache apptainer build --fakeroot \\" >&2
    echo "    $AGENT_SIF $ROOT/agents/apptainer/pi/agent.def" >&2
    exit 1
fi

# WORK_ROOT goes on node-local SSD via \$TMPDIR (set by SLURM); mktemp honors it.
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
        # Emit provider-qualified ids (provider/id). pi 0.80.2 resolves a bare id
        # like "DeepSeek-V4-Flash" against its built-in catalog first (-> wrong
        # provider, "No API key"), so the provider prefix is required to bind the
        # model to our custom providers in models.json.
        EVAL_MODELS=$(jq -r '.providers | to_entries[] | .key as $p | .value.models[] | "\($p)/\(.id)"' "$MODELS_CONFIG" | paste -sd, -)
    else
        echo "ERROR: EVAL_MODELS is not set and jq is unavailable to parse $MODELS_CONFIG" >&2
        exit 1
    fi
fi

IFS=',' read -r -a MODELS <<< "$EVAL_MODELS"
IFS=',' read -r -a PROBLEMS <<< "$EVAL_PROBLEMS"

# Deterministic per-job paths, shared by both phases (no cross-phase state).
job_workdir()  { echo "$WORK_ROOT/work/$1/round-$2/$3"; }                   # MODEL_TAG ROUND PROBLEM_NAME
job_home()     { echo "$WORK_ROOT/home/$1/round-$2/$3"; }                   # pi HOME staging (holds models.json)
job_tracedir() { echo "$TRACE_ROOT/$3/$AGENT/$1/$DATE_TAG/round-$2"; }      # MODEL_TAG ROUND PROBLEM_NAME

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
# Phase 1 (SOLVE): run the agent. Up to EVAL_SOLVE_PAR at once, sharing the GPU.
# ---------------------------------------------------------------------------
solve_job() {
    local MODEL="$1" ROUND="$2" PROBLEM_NAME="$3"
    local MODEL_TAG="${MODEL//\//_}"
    local PROBLEM="$PROBLEM_DIR/$PROBLEM_NAME"
    local WORK_DIR; WORK_DIR="$(job_workdir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
    local PIHOME;   PIHOME="$(job_home "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"

    echo "[solve] problem: $PROBLEM_NAME | agent: $AGENT | model: $MODEL | round: $ROUND"
    mkdir -p "$WORK_DIR"
    # rsync, not `cp -a`: cp -a tries to replicate ACLs, and on the test-cluster
    # filesystem that fails with "preserving permissions ... Operation not
    # supported" for every single file — ~25 lines of noise per round, drowning
    # the log. rsync -a preserves what actually matters here (mode, mtime,
    # symlinks) and does not touch ACLs unless asked with -A. mtime matters: the
    # `rsync --compare-dest` in measure_job decides "did the agent modify this
    # file?" from size+mtime, so a copy that reset mtimes would make every file
    # look tampered with.
    rsync -a "$PROBLEM"/ "$WORK_DIR"/

    # pi reads ~/.pi/agent/models.json. Apptainer runs as you with HOME=your real
    # home, so stage a clean HOME holding the config and bind it to /root.
    mkdir -p "$PIHOME/.pi/agent"
    cp "$MODELS_CONFIG" "$PIHOME/.pi/agent/models.json"

    local DATA_BIND=()
    if [[ -d "$DATA_ROOT/$PROBLEM_NAME/RabbitInput" ]]; then
        DATA_BIND=(--bind "$DATA_ROOT/$PROBLEM_NAME/RabbitInput:/problem/RabbitInput:ro")
    fi

    apptainer exec --cleanenv "${NV_FLAG[@]}" \
        --home "$PIHOME:/root" \
        --env "EVAL_AGENT=$AGENT" \
        --env "EVAL_MODEL=$MODEL" \
        --env "EVAL_ROUND=$ROUND" \
        --env "EVAL_SIZE=$EVAL_SIZE" \
        --env "EVAL_VARIANT=$EVAL_VARIANT" \
        --env "EVAL_STEP_LIMIT=$EVAL_STEP_LIMIT" \
        --bind "$WORK_DIR:/problem" \
        --bind "$SOLVE_SCRIPT:/solve.sh:ro" \
        "${DATA_BIND[@]}" \
        "$AGENT_SIF" /bin/bash -l /solve.sh /problem
}

# ---------------------------------------------------------------------------
# Phase 2 (MEASURE): replay build/run with GPU sampling, then extract the trace.
# Strictly serial so each graded run owns the GPU.
# ---------------------------------------------------------------------------
measure_job() {
    local MODEL="$1" ROUND="$2" PROBLEM_NAME="$3"
    local MODEL_TAG="${MODEL//\//_}"
    local PROBLEM="$PROBLEM_DIR/$PROBLEM_NAME"
    local WORK_DIR;  WORK_DIR="$(job_workdir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
    local PIHOME;    PIHOME="$(job_home "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
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

    # Host-side trace extraction (no container needed — Apptainer wrote everything
    # as you on a coherent FS). --compare-dest keeps only files the agent changed
    # or added (sources, solution.json, agent.jsonl, .eval/).
    mkdir -p "$TRACE_DIR"
    rsync -a --prune-empty-dirs --compare-dest="$PROBLEM/" --exclude=".pi/agent/" "$WORK_DIR/" "$TRACE_DIR/"
    diff -ru --exclude=".git" "$PROBLEM/" "$WORK_DIR/" > "$TRACE_DIR/changed_diff.txt" 2>/dev/null || true

    if [ ! -s "$TRACE_DIR/.eval/facts.json" ]; then
        echo "ERROR: no .eval/facts.json in trace for ${PROBLEM_NAME} | ${MODEL} | round-${ROUND} — extraction produced no evidence. Aborting eval." >&2
        exit 1
    fi
    cat "$TRACE_DIR/changed_diff.txt" 2>/dev/null || true

    # Trace is safely captured — free this job's workdir + home.
    rm -rf "$WORK_DIR" "$PIHOME"
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

# Host-side pass judgment (no GPU, re-runnable).
if command -v python3 >/dev/null 2>&1; then
    python3 "$ROOT/checks/check_pass.py" --traces "$TRACE_ROOT"
    python3 "$ROOT/checks/trace_stats.py" --traces "$TRACE_ROOT"
else
    echo "WARN: python3 not found on host — skipping pass checks. Run checks/check_pass.py + trace_stats.py manually." >&2
fi
