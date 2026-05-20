#! /bin/bash -l

ROOT=/home/horatio/projects/nhr-agent-evaluation
PROBLEM_DIR="$ROOT/problems"
TRACE_ROOT="$ROOT/traces"
DOCKERFILE="$ROOT/agents/docker/pi/Dockerfile"
IMAGE="agent-eval-pi:latest"
CONTAINER_SCRIPT="$ROOT/agents/docker/pi/guest.sh"
MODELS_CONFIG="$ROOT/agents/apptainer/pi/models.json"

DATE_TAG=$(date +%Y%m%d-%H%M%S)

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -t "$IMAGE" -f "$DOCKERFILE" "$(dirname "$DOCKERFILE")"
fi

WORK_ROOT=$(mktemp -d)
trap 'rm -rf "$WORK_ROOT"' EXIT

# for PROBLEM in "$PROBLEM_DIR"/*; do
PROBLEM="$PROBLEM_DIR/MD-Bench" # for testing
    if [[ ! -d "$PROBLEM" ]]; then
        continue
    fi

    PROBLEM_NAME=$(basename "$PROBLEM")
    TRACE_DIR="$TRACE_ROOT/$PROBLEM_NAME/pi/$DATE_TAG"
    WORK_DIR="$WORK_ROOT/$PROBLEM_NAME"

    echo "Evaluating problem: $PROBLEM_NAME"
    mkdir -p "$WORK_DIR"
    cp -a "$PROBLEM"/. "$WORK_DIR"/

    docker run --rm \
        -v "$WORK_DIR:/problem" \
        -v "$MODELS_CONFIG:/root/.pi/agent/models.json:ro" \
        -v "$CONTAINER_SCRIPT:/host.sh:ro" \
        "$IMAGE" /bin/bash -l /host.sh /problem

    mkdir -p "$TRACE_DIR"
    TEMPDIR=$(mktemp -d)
    # first creates empty folders (sadly)
    rsync -a --prune-empty-dirs \
        --compare-dest="$PROBLEM/" \
        --exclude='.pi/agent/' \
        "$WORK_DIR/" "$TEMPDIR/"
    # second skips empty folders the changed files
    rsync -a --prune-empty-dirs \
        --compare-dest="$PROBLEM/" \
        --exclude='.pi/agent/' \
        "$TEMPDIR/" "$TRACE_DIR/"
    
    diff -ru --exclude='.git' --color=always "$PROBLEM/" "$WORK_DIR/" | tee "$TRACE_DIR/changed_diff.txt"
# done
