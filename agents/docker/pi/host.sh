#! /bin/bash -l

ROOT="$(pwd)"
PROBLEM_DIR="$ROOT/problems"
TRACE_ROOT="$ROOT/traces"
DOCKERFILE="$ROOT/agents/docker/pi/Dockerfile"
IMAGE="agent-eval-pi:latest"
CONTAINER_SCRIPT="$ROOT/agents/docker/pi/guest.sh"
MODELS_CONFIG="${MODELS_CONFIG:-$ROOT/agents/apptainer/pi/models.json}"
EVAL_ROUNDS="${EVAL_ROUNDS:-20}"
EVAL_PROBLEMS="${EVAL_PROBLEMS:-RabbitCT}"
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

            TRACE_DIR="$TRACE_ROOT/$PROBLEM_NAME/$MODEL_TAG/$DATE_TAG/round-$ROUND"
            WORK_DIR="$WORK_ROOT/$PROBLEM_NAME"

            echo "Evaluating problem: $PROBLEM_NAME | model: $MODEL | round: $ROUND"
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
                -e "EVAL_MODEL=$MODEL" \
                -e "EVAL_ROUND=$ROUND" \
                -e "EVAL_SIZE=$EVAL_SIZE" \
                -e "EVAL_VARIANT=$EVAL_VARIANT" \
                -v "$WORK_DIR:/problem" \
                -v "$MODELS_CONFIG:/root/.pi/agent/models.json:ro" \
                -v "$CONTAINER_SCRIPT:/host.sh:ro" \
                "${DATA_MOUNT[@]}" \
                "$IMAGE" /bin/bash -l /host.sh /problem

            # The eval container runs as root, so everything it wrote into the
            # bind-mounted workdir (.eval/, build/, solution.json, ...) is owned
            # by root:root. Hand it back to the invoking user via a throwaway
            # root container. Two reasons this is needed:
            #   1. The EXIT-trap cleanup (rm -rf "$WORK_ROOT") runs as this user
            #      and otherwise can't delete root-owned files -> "Permission
            #      denied" spam and leftover /tmp dirs.
            #   2. It re-touches every file through a container write, forcing
            #      the bind mount to flush so the host-side rsync below reliably
            #      sees the just-written .eval/ evidence (otherwise traces/ can
            #      end up with only changed_diff.txt and check_pass finds nothing).
            docker run --rm \
                -v "$WORK_DIR:/problem" \
                "$IMAGE" chown -R "$(id -u):$(id -g)" /problem

            mkdir -p "$TRACE_DIR"
            TEMPDIR=$(mktemp -d)
            rsync -a --prune-empty-dirs \
                --compare-dest="$PROBLEM/" \
                --exclude='.pi/agent/' \
                "$WORK_DIR/" "$TEMPDIR/"
            rsync -a --prune-empty-dirs \
                --compare-dest="$PROBLEM/" \
                --exclude='.pi/agent/' \
                "$TEMPDIR/" "$TRACE_DIR/"

            diff -ru --exclude='.git' --color=always "$PROBLEM/" "$WORK_DIR/" | tee "$TRACE_DIR/changed_diff.txt"
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
