#! /bin/bash -l

# swe-mini agent orchestrator. Mirror of agents/docker/pi/host.sh; differs only
# in the agent-specific bits (image, guest script, config mounts, AGENT tag).
# Keep the shared logic (problem copy, data mount, chown-back, evidence
# visibility wait, rsync, diff, check_pass) in sync with the pi version.

AGENT="swe-mini"
ROOT="$(pwd)"
PROBLEM_DIR="$ROOT/problems"
TRACE_ROOT="$ROOT/traces"
DOCKERFILE="$ROOT/agents/docker/swe-mini/Dockerfile"
IMAGE="agent-eval-swe-mini:latest"
CONTAINER_SCRIPT="$ROOT/agents/docker/swe-mini/guest.sh"
MINI_CONFIG="$ROOT/agents/docker/swe-mini/mini.yaml"
MODELS_CONFIG="${MODELS_CONFIG:-$ROOT/agents/apptainer/pi/models.json}"
EVAL_ROUNDS="${EVAL_ROUNDS:-20}"
EVAL_PROBLEMS="${EVAL_PROBLEMS:-RabbitCT-LolaBunny}"
EVAL_SIZE="${EVAL_SIZE:-256}"
EVAL_VARIANT="${EVAL_VARIANT:-LolaBunnyGPU}"
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

for MODEL in "${MODELS[@]}"; do
    MODEL_TAG=${MODEL//\//_}
    for ROUND in $(seq 1 "$EVAL_ROUNDS"); do
        for PROBLEM_NAME in "${PROBLEMS[@]}"; do
            PROBLEM="$PROBLEM_DIR/$PROBLEM_NAME"
            if [[ ! -d "$PROBLEM" ]]; then
                echo "Skipping missing problem: $PROBLEM" >&2
                continue
            fi

            # Trace layout includes the agent segment AFTER the problem so pi and
            # swe-mini runs don't collide and check_pass.py (problem = parts[0])
            # still works unchanged.
            TRACE_DIR="$TRACE_ROOT/$PROBLEM_NAME/$AGENT/$MODEL_TAG/$DATE_TAG/round-$ROUND"
            WORK_DIR="$WORK_ROOT/$PROBLEM_NAME"

            echo "Evaluating problem: $PROBLEM_NAME | agent: $AGENT | model: $MODEL | round: $ROUND"
            mkdir -p "$WORK_DIR"
            rm -rf "$WORK_DIR"/*
            cp -a "$PROBLEM"/. "$WORK_DIR"/

            # Read-only mount for problems whose bulk input data lives in
            # $DATA_ROOT/<problem>/ instead of the copied workdir (e.g. RabbitCT).
            # Mounted at the path the problem expects; the empty mountpoint left
            # in the workdir is pruned by the rsync below. Empty array = no-op for
            # problems without external data.
            DATA_MOUNT=()
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
                -v "$CONTAINER_SCRIPT:/host.sh:ro" \
                "${DATA_MOUNT[@]}" \
                "$IMAGE" /bin/bash -l /host.sh /problem

            # ----------------------------------------------------------------
            # Extract the trace from INSIDE a container, streamed out as a tar
            # over stdout. Why not host-side rsync/diff: the eval container ran
            # as root, and on WSL2 the host's view of the bind mount lags the
            # container's writes — a path stat of .eval/facts.json can succeed
            # while a readdir of the workdir still omits that (newest) entry, and
            # rsync/diff discover files via readdir, so they intermittently
            # produced .eval-less, diff-only traces (a poll that only stats the
            # file does not fix this). A container's view of the just-written
            # workdir IS coherent, so we do the rsync/diff there and pipe the
            # result out as a tar; the host only WRITES TRACE_DIR from that byte
            # stream, so its own readdir is always coherent. The same container
            # also chowns the workdir back so the EXIT-trap cleanup (as this
            # user) can delete it. --entrypoint bash skips the CUDA image banner
            # so stdout is pure tar; rsync/chown are quiet and diff is redirected
            # to a file, so nothing else pollutes the stream.
            # ----------------------------------------------------------------
            mkdir -p "$TRACE_DIR"
            TRACE_TAR="$WORK_ROOT/trace-${PROBLEM_NAME}-${ROUND}.tar"
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

            # Fail LOUD if the agent produced no judgable evidence. guest.sh
            # always writes a .eval/facts.json sentinel (even on agent failure),
            # so its absence means extraction genuinely failed. This reads what
            # we just wrote on the host (coherent), not the bind mount.
            if [ ! -s "$TRACE_DIR/.eval/facts.json" ]; then
                echo "ERROR: no .eval/facts.json in trace for ${PROBLEM_NAME} | ${MODEL} | round-${ROUND} — extraction produced no evidence. Aborting eval." >&2
                exit 1
            fi
            cat "$TRACE_DIR/changed_diff.txt" 2>/dev/null || true
        done
    done
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
