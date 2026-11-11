#! /bin/bash -l

# claude (Claude Code) agent orchestrator — APPTAINER runtime (cluster / SLURM).
# Two-phase mirror of agents/apptainer/swe-mini/host.sh (see the pi
# host.sh for the shared structure).
#
# claude specifics vs swe-mini: the agent is Claude Code, configured purely via
# env (set in this dir's solve.sh) so EVAL_MODELS works at submit time — no
# mini.yaml. It still reads the gateway baseUrl + apiKey from /models.json
# (bind-mounted) and gets a clean staging HOME so its ~/.claude state doesn't
# land in your real home. The in-container measure.sh is agent-agnostic and is
# the single shared copy at agents/measure.sh.
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
MEASURE_SCRIPT="$ROOT/agents/measure.sh"   # agent-agnostic, shared by every agent
# Endpoint config: endpoints.yaml + .env -> .run/models.json, EVAL_PROVIDER, EVAL_MODELS
# (scripts/lib/endpoint.sh). Skipped when MODELS_CONFIG is already set.
. "$ROOT/scripts/lib/endpoint.sh" && resolve_endpoints "$ROOT" claude || exit 1
EVAL_ROUNDS="${EVAL_ROUNDS:-20}"
EVAL_PROBLEMS="${EVAL_PROBLEMS:-RabbitCT-LolaBunny}"
EVAL_SIZE="${EVAL_SIZE:-256}"
EVAL_VARIANT="${EVAL_VARIANT:-LolaBunnyGPU}"
EVAL_SOLVE_PAR="${EVAL_SOLVE_PAR:-4}"
EVAL_MAX_TURNS="${EVAL_MAX_TURNS:-120}"
# Reasoning effort (`claude --effort`). Kept identical to claude-pro so the
# gateway-vs-subscription comparison differs only in auth/model; expected to be a
# no-op for non-Anthropic gateway models (see solve.sh).
EVAL_EFFORT="${EVAL_EFFORT:-high}"
# providers.<name> in models.json that supplies baseUrl + apiKey. Set by
# scripts/lib/resolve_endpoint.py (= the endpoint name); "local" only for a hand-made
# models.json. Forwarded via --env below: apptainer exec runs with --cleanenv, so
# anything needed inside must be forwarded explicitly.
EVAL_PROVIDER="${EVAL_PROVIDER:-local}"
# Claude Code token knobs (consumed by solve.sh). Empty = use solve.sh's own
# default (output cap 64000; thinking unset = max). MUST be forwarded explicitly
# via --env below. NB: apptainer exec inherits the host environment by DEFAULT; we pass --cleanenv
#   to block that, so anything needed inside must be forwarded explicitly.
EVAL_MAX_OUTPUT_TOKENS="${EVAL_MAX_OUTPUT_TOKENS:-}"
EVAL_MAX_THINKING_TOKENS="${EVAL_MAX_THINKING_TOKENS:-}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
# GPU-exclusive lock for Phase 2 (measure) + the trailing global stats regen.
# Solve is contention-tolerant (see solve.sh), but multiple campaigns can now
# share one node/GPU (packed run) — measure needs real isolation for valid
# timing, and the stats scripts write shared stats.json files across the whole
# traces/ tree. Keyed off $TMPDIR so it's shared by every host.sh instance
# packed into the same SLURM job, and harmless (single-process) otherwise.
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
    echo "    $AGENT_SIF $ROOT/agents/apptainer/claude/agent.def" >&2
    exit 1
fi

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

solve_job() {
    local MODEL="$1" ROUND="$2" PROBLEM_NAME="$3"
    local MODEL_TAG="${EVAL_MODEL_TAG:-${MODEL//\//_}}"
    local PROBLEM="$PROBLEM_DIR/$PROBLEM_NAME"
    local WORK_DIR; WORK_DIR="$(job_workdir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
    local CLAUDEHOME; CLAUDEHOME="$(job_home "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
    local TRACE_DIR; TRACE_DIR="$(job_tracedir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"

    echo "[solve] problem: $PROBLEM_NAME | agent: $AGENT | model: $MODEL | round: $ROUND"
    mkdir -p "$WORK_DIR" "$CLAUDEHOME"
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

    local LOG_FILE="$WORK_DIR/agent.stream.jsonl"

    # --pid (2026-09-24): own PID namespace per round. Every process the agent started
    # dies with the container, on a normal end and on a timeout kill (orphaned find
    # after a kimi timeout), and an agent pkill cannot reach sibling rounds.
    apptainer exec --cleanenv --pid "${NV_FLAG[@]}" \
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
        --env "EVAL_EFFORT=$EVAL_EFFORT" \
        --bind "$WORK_DIR:/problem" \
        --bind "$MODELS_CONFIG:/models.json:ro" \
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
    rsync -a --prune-empty-dirs --compare-dest="$PROBLEM/" "$WORK_DIR/" "$TRACE_DIR/"
    diff -ru --exclude=".git" "$PROBLEM/" "$WORK_DIR/" > "$TRACE_DIR/changed_diff.txt" 2>/dev/null || true

    if [ "$rc" -eq 124 ]; then
        echo "[solve] TIMEOUT (${kill_reason:-external}): $PROBLEM_NAME | round: $ROUND -- partial output evacuated to $TRACE_DIR, skipping measure" >&2
        touch "$TRACE_DIR/.solve_timeout"
    fi
    return "$rc"
}

measure_job() {
    local MODEL="$1" ROUND="$2" PROBLEM_NAME="$3"
    local MODEL_TAG="${EVAL_MODEL_TAG:-${MODEL//\//_}}"
    local PROBLEM="$PROBLEM_DIR/$PROBLEM_NAME"
    local WORK_DIR;  WORK_DIR="$(job_workdir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
    local CLAUDEHOME; CLAUDEHOME="$(job_home "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"
    local TRACE_DIR; TRACE_DIR="$(job_tracedir "$MODEL_TAG" "$ROUND" "$PROBLEM_NAME")"

    if [ -f "$TRACE_DIR/.solve_timeout" ]; then
        echo "[measure] SKIP $PROBLEM_NAME | round: $ROUND -- solve phase timed out, no complete solution to measure" >&2
        rm -rf "$WORK_DIR" "$CLAUDEHOME"
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

    mkdir -p "$TRACE_DIR"
    rsync -a --prune-empty-dirs --compare-dest="$PROBLEM/" "$WORK_DIR/" "$TRACE_DIR/"
    diff -ru --exclude=".git" "$PROBLEM/" "$WORK_DIR/" > "$TRACE_DIR/changed_diff.txt" 2>/dev/null || true

    if [ "$rc" -eq 124 ]; then
        echo "[measure] TIMEOUT after ${EVAL_MEASURE_TIMEOUT}s: $PROBLEM_NAME | round: $ROUND -- solve output already evacuated at $TRACE_DIR (no .eval/facts.json); use scripts/remeasure.sh to retry measurement standalone" >&2
        touch "$TRACE_DIR/.measure_timeout"
    fi

    if [ ! -s "$TRACE_DIR/.eval/facts.json" ]; then
        echo "ERROR: no .eval/facts.json in trace for ${PROBLEM_NAME} | ${MODEL} | round-${ROUND} — extraction produced no evidence." >&2
        if [ "$rc" -ne 124 ]; then
            rm -rf "$WORK_DIR" "$CLAUDEHOME"
            return 1
        fi
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
