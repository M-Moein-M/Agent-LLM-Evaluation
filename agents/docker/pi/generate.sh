#! /bin/bash -l

ROOT=/home/horatio/projects/nhr-agent-evaluation
PROBLEM_DIR="$ROOT/problems"
DOCKERFILE="$ROOT/agents/docker/pi/Dockerfile"
IMAGE="agent-eval-pi:latest"
CONTAINER_SCRIPT="$ROOT/agents/docker/pi/guest.sh"
MODELS_CONFIG="$ROOT/agents/apptainer/pi/models.json"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -t "$IMAGE" -f "$DOCKERFILE" "$(dirname "$DOCKERFILE")"
fi

EVAL_TMPDIR=$(mktemp -d)
trap 'rm -rf "$EVAL_TMPDIR"' EXIT
cp -r "$PROBLEM_DIR"/* "$EVAL_TMPDIR"/

for PROBLEM in "$EVAL_TMPDIR"/*; do
    if [[ ! -d "$PROBLEM" ]]; then
        continue
    fi

    echo "Evaluating problem: $PROBLEM"
    mkdir -p "$PROBLEM/.pi/agent"
    cp "$MODELS_CONFIG" "$PROBLEM/.pi/agent/"

    docker run --rm \
        -v "$PROBLEM:/problem" \
        -v "$MODELS_CONFIG:/models.json:ro" \
        -v "$CONTAINER_SCRIPT:/host.sh:ro" \
        "$IMAGE" /bin/bash -l /host.sh /problem
done
