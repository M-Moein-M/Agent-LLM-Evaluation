#! /bin/bash -l

# pi agent orchestrator (Apptainer, SLURM or a plain node). Two phases: parallel solve,
# serial measure. The other agents' host.sh files mirror this one; keep shared logic in step.
#   * the image is a prebuilt .sif (apptainer build --fakeroot ... agent.def)
#   * pi reads ~/.pi/agent/models.json: Apptainer runs as YOU with your real HOME, so a
#     per-job staging dir holding models.json is bound over /root (--home)
#   * trace extraction is a plain host-side rsync/diff
# The in-container scripts solve.sh (this dir) and agents/measure.sh run inside the container.

AGENT="pi"
ROOT="$(pwd)"
PROBLEM_DIR="$ROOT/problems"
TRACE_ROOT="$ROOT/traces"
AGENT_SIF="${AGENT_SIF:-$ROOT/agents/apptainer/pi/agent.sif}"
# In-container scripts, bound into the container.
SOLVE_SCRIPT="$ROOT/agents/apptainer/pi/solve.sh"
MEASURE_SCRIPT="$ROOT/agents/measure.sh"
# Endpoint config: endpoints.yaml + .env -> .run/models.json, EVAL_PROVIDER, EVAL_MODELS
# (scripts/lib/endpoint.sh). Skipped when MODELS_CONFIG is already set.
. "$ROOT/scripts/lib/endpoint.sh" && resolve_endpoints "$ROOT" pi || exit 1
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
# Cross-process GPU serialization for Phase 2. Measure is already serial WITHIN
# one host.sh run; this lock matters when multiple host.sh processes (e.g. a
# different agent, or another model's wave) share one GPU inside the same SLURM
# job — mirrors agents/apptainer/claude/host.sh. Default path is per-job
# ($TMPDIR), so unrelated jobs never contend; set MEASURE_LOCK explicitly to
# share it across processes within one job (e.g. a merged kimi+pi wave).
MEASURE_LOCK="${MEASURE_LOCK:-${TMPDIR:-/tmp}/agent-eval-measure.lock}"
# Solve-phase liveness watchdog (see agents/apptainer/kimi/host.sh for the
# full rationale/job history -- a flat 30min EVAL_SOLVE_TIMEOUT was killing
# legitimate slow rounds mid-compile/mid-benchmark). solve_job backgrounds
# apptainer exec and polls the agent's own trace file (size+mtime) instead of
# blindly capping wall time:
#   EVAL_SOLVE_STALL_TIMEOUT -- kill if the log file hasn't changed in this
#     long ("dead vs progress"). 1h default.
#   EVAL_SOLVE_TIMEOUT -- absolute backstop per round regardless of liveness.
#     Bumped from the old 30min flat cap to 4h now that liveness, not wall
#     time, is the primary kill signal.
#   EVAL_SOLVE_POLL_INTERVAL -- how often to check.
# Each parallel Phase 1 solve_job() call already runs as its own backgrounded
# subshell, so concurrent rounds get independent timers for free.
EVAL_SOLVE_STALL_TIMEOUT="${EVAL_SOLVE_STALL_TIMEOUT:-3600}"
EVAL_SOLVE_TIMEOUT="${EVAL_SOLVE_TIMEOUT:-14400}"
EVAL_SOLVE_POLL_INTERVAL="${EVAL_SOLVE_POLL_INTERVAL:-60}"
# Measure phase stays a flat blocking timeout -- strictly serial +
# GPU-isolated already, a different failure mode from solve's LLM-request
# stalls, out of scope for this fix.
EVAL_MEASURE_TIMEOUT="${EVAL_MEASURE_TIMEOUT:-1800}"
# Overrides the trace/workdir model folder name, which is otherwise derived from
# the model id sent to the gateway. Needed when the gateway REUSES one id for a
# new checkpoint (e.g. deepseek-ai/DeepSeek-V4-Flash silently became the -0731
# build on 2026-08-10): the wire id must stay the served one, but its traces have
# to land under a distinct name so the two checkpoints stay comparable instead of
# being merged into one directory. Single-model runs ONLY — with several ids in
# EVAL_MODELS they would all collapse into this one folder.
EVAL_MODEL_TAG="${EVAL_MODEL_TAG:-}"

# EVAL_DATE_TAG / EVAL_ROUND_IDS: fixed campaign dir and round numbers, so one-round
# calls from scripts/phase6/worker.py land in one campaign per cell.
DATE_TAG="${EVAL_DATE_TAG:-$(date +%Y%m%d-%H%M%S)}"
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
# (e.g. on a node without a GPU), where --nv on a node without an NVIDIA
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
    for ROUND in ${EVAL_ROUND_IDS:-$(seq 1 "$EVAL_ROUNDS")}; do
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
    local MODEL_TAG="${EVAL_MODEL_TAG:-${MODEL//\//_}}"
    local PROBLEM="$PROBLEM_DIR/$PROBLEM_NAME"
    local WORK_DIR; WORK_DIR="$(job_workdir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
    local PIHOME;   PIHOME="$(job_home "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
    local TRACE_DIR; TRACE_DIR="$(job_tracedir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"

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
    # Unified limits (2026-09-16): no compaction (settings-manager.ts compaction.enabled);
    # a server context-overflow ends the run like swe-mini.
    # O3 (2026-09-18): fake streaming sends nothing until a reply is complete. pi's HTTP idle
    # timeout (default 5 min, http-dispatcher.ts) and SDK request timeout (sdk.ts, falls back
    # to the idle value) would end long replies and pi would retry them. Both 4h.
    echo '{"compaction": {"enabled": false}, "httpIdleTimeoutMs": 14400000, "retry": {"provider": {"timeoutMs": 14400000}}}' > "$PIHOME/.pi/agent/settings.json"

    local DATA_BIND=()
    if [[ -d "$DATA_ROOT/$PROBLEM_NAME/RabbitInput" ]]; then
        DATA_BIND=(--bind "$DATA_ROOT/$PROBLEM_NAME/RabbitInput:/problem/RabbitInput:ro")
    fi

    local LOG_FILE="$WORK_DIR/agent.jsonl"

    # --pid (2026-09-24): own PID namespace per round. Every process the agent started
    # dies with the container, on a normal end and on a timeout kill (orphaned find
    # after a kimi timeout), and an agent pkill cannot reach sibling rounds.
    apptainer exec --cleanenv --pid "${NV_FLAG[@]}" \
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
        "$AGENT_SIF" /bin/bash -l /solve.sh /problem &
    local APP_PID=$!

    # Liveness watchdog -- see EVAL_SOLVE_STALL_TIMEOUT/EVAL_SOLVE_TIMEOUT
    # comment near the top of this file.
    local START_TS=$SECONDS last_size=-1 last_mtime=-1 last_change_ts=$SECONDS kill_reason=""
    while kill -0 "$APP_PID" 2>/dev/null; do
        sleep "$EVAL_SOLVE_POLL_INTERVAL"
        if [[ -f "$LOG_FILE" ]]; then
            local cur_size cur_mtime
            cur_size=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo -1)
            cur_mtime=$(stat -c%Y "$LOG_FILE" 2>/dev/null || echo -1)
            if [[ "$cur_size" != "$last_size" || "$cur_mtime" != "$last_mtime" ]]; then
                last_size="$cur_size"; last_mtime="$cur_mtime"; last_change_ts=$SECONDS
            fi
        fi
        if (( SECONDS - last_change_ts >= EVAL_SOLVE_STALL_TIMEOUT )); then
            kill_reason="STALL: no change to $(basename "$LOG_FILE") for ${EVAL_SOLVE_STALL_TIMEOUT}s"
            break
        fi
        if (( SECONDS - START_TS >= EVAL_SOLVE_TIMEOUT )); then
            kill_reason="ABSOLUTE TIMEOUT after ${EVAL_SOLVE_TIMEOUT}s"
            break
        fi
    done

    local rc
    if [[ -n "$kill_reason" ]]; then
        echo "[solve] $kill_reason: $PROBLEM_NAME | round: $ROUND -- killing pid $APP_PID" >&2
        kill -TERM "$APP_PID" 2>/dev/null
        for _ in $(seq 1 10); do kill -0 "$APP_PID" 2>/dev/null || break; sleep 1; done
        kill -KILL "$APP_PID" 2>/dev/null
        wait "$APP_PID" 2>/dev/null
        rc=124
    else
        wait "$APP_PID"
        rc=$?
    fi

    mkdir -p "$TRACE_DIR"
    rsync -a --prune-empty-dirs --compare-dest="$PROBLEM/" --exclude=".pi/agent/" "$WORK_DIR/" "$TRACE_DIR/"
    diff -ru --exclude=".git" "$PROBLEM/" "$WORK_DIR/" > "$TRACE_DIR/changed_diff.txt" 2>/dev/null || true

    if [ "$rc" -eq 124 ]; then
        echo "[solve] TIMEOUT (${kill_reason:-external}): $PROBLEM_NAME | round: $ROUND -- partial output evacuated to $TRACE_DIR, skipping measure" >&2
        touch "$TRACE_DIR/.solve_timeout"
    elif [ "$rc" -ne 0 ]; then
        echo "[solve] FAILED rc=$rc: $PROBLEM_NAME | round: $ROUND -- solve.sh exited non-zero, output evacuated to $TRACE_DIR" >&2
    fi

    # agent.jsonl is the one artifact nothing downstream can reconstruct. Phase 1
    # backgrounds solve_job with `&` and never reads the status off `wait -n`, and
    # measure_job still grades whatever sources the agent left in the workdir — so
    # a solve that died before persisting its trace produces a complete-looking,
    # fully measured round with no log. Mark it at the point we still know.
    if [ ! -s "$TRACE_DIR/agent.jsonl" ]; then
        echo "[solve] NO TRACE: $PROBLEM_NAME | round: $ROUND (rc=$rc) -- no agent.jsonl in $TRACE_DIR" >&2
        touch "$TRACE_DIR/.no_trace"
    fi
    return "$rc"
}

# ---------------------------------------------------------------------------
# Phase 2 (MEASURE): replay build/run with GPU sampling, then extract the trace.
# Strictly serial so each graded run owns the GPU.
# ---------------------------------------------------------------------------
measure_job() {
    local MODEL="$1" ROUND="$2" PROBLEM_NAME="$3"
    local MODEL_TAG="${EVAL_MODEL_TAG:-${MODEL//\//_}}"
    local PROBLEM="$PROBLEM_DIR/$PROBLEM_NAME"
    local WORK_DIR;  WORK_DIR="$(job_workdir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
    local PIHOME;    PIHOME="$(job_home "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
    local TRACE_DIR; TRACE_DIR="$(job_tracedir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"

    if [ -f "$TRACE_DIR/.solve_timeout" ]; then
        echo "[measure] SKIP $PROBLEM_NAME | round: $ROUND -- solve phase timed out, no complete solution to measure" >&2
        rm -rf "$WORK_DIR" "$PIHOME"
        return 0
    fi

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

    (
        flock -x 200
        timeout "$EVAL_MEASURE_TIMEOUT" apptainer exec --cleanenv --pid "${NV_FLAG[@]}" \
            --env "EVAL_AGENT=$AGENT" \
            --env "EVAL_MODEL=$MODEL" \
            --env "EVAL_ROUND=$ROUND" \
            --env "EVAL_SIZE=$EVAL_SIZE" \
            --env "EVAL_VARIANT=$EVAL_VARIANT" \
            --env "EVAL_REQUIRE_GPU=${EVAL_REQUIRE_GPU:-0}" \
            --bind "$WORK_DIR:/problem" \
            --bind "$MEASURE_SCRIPT:/measure.sh:ro" \
            "${DATA_BIND[@]}" \
            "${GRADE_BIND[@]}" \
            "$AGENT_SIF" /bin/bash -l /measure.sh /problem
    ) 200>"$MEASURE_LOCK"
    local rc=$?

    # Host-side trace extraction (no container needed — Apptainer wrote everything
    # as you on a coherent FS). --compare-dest keeps only files the agent changed
    # or added (sources, solution.json, agent.jsonl, .eval/).
    mkdir -p "$TRACE_DIR"
    rsync -a --prune-empty-dirs --compare-dest="$PROBLEM/" --exclude=".pi/agent/" "$WORK_DIR/" "$TRACE_DIR/"
    diff -ru --exclude=".git" "$PROBLEM/" "$WORK_DIR/" > "$TRACE_DIR/changed_diff.txt" 2>/dev/null || true

    if [ "$rc" -eq 124 ]; then
        echo "[measure] TIMEOUT after ${EVAL_MEASURE_TIMEOUT}s: $PROBLEM_NAME | round: $ROUND -- solve output already evacuated at $TRACE_DIR (no .eval/facts.json); use scripts/remeasure.sh to retry measurement standalone" >&2
        touch "$TRACE_DIR/.measure_timeout"
    fi

    if [ ! -s "$TRACE_DIR/.eval/facts.json" ]; then
        echo "ERROR: no .eval/facts.json in trace for ${PROBLEM_NAME} | ${MODEL} | round-${ROUND} — extraction produced no evidence." >&2
        if [ "$rc" -ne 124 ]; then
            rm -rf "$WORK_DIR" "$PIHOME"
            return 1
        fi
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
if [ "${EVAL_SKIP_CHECKS:-0}" = "1" ]; then
    echo "EVAL_SKIP_CHECKS=1: skipping check_pass.py + trace_stats.py (caller runs them)."
elif command -v python3 >/dev/null 2>&1; then
    (
        flock -x 200
        python3 "$ROOT/checks/check_pass.py" --traces "$TRACE_ROOT" --campaign "$DATE_TAG"
        python3 "$ROOT/checks/trace_stats.py" --traces "$TRACE_ROOT" --campaign "$DATE_TAG"
    ) 200>"$MEASURE_LOCK"
else
    echo "WARN: python3 not found on host — skipping pass checks. Run checks/check_pass.py + trace_stats.py manually." >&2
fi
