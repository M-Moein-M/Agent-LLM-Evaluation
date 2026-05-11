#! /bin/bash -l

ROOT=/home/${GROUP}/${USER}/agent-evaluation-nhr
PROBLEM_DIR="$ROOT/problems"
TRACE_ROOT="$ROOT/traces"
AGENT_SIF="$ROOT/agents/claude/agent.sif"
SETTINGS_JSON="$ROOT/agents/claude/settings.json"
CONTAINER_SCRIPT="$ROOT/agents/claude/guest.sh"

DATE_TAG=$(date +%Y%m%d-%H%M%S)

for PROBLEM in "$PROBLEM_DIR"/*; do
    if [[ ! -d "$PROBLEM" ]]; then
        continue
    fi

    PROBLEM_NAME=$(basename "$PROBLEM")
    TRACE_DIR="$TRACE_ROOT/$PROBLEM_NAME/claude/$DATE_TAG"
    PROBLEM_COPY="$TRACE_DIR/$PROBLEM_NAME"

    echo "Evaluating problem: $PROBLEM_NAME"
    mkdir -p "$TRACE_DIR"
    cp -r "$PROBLEM" "$PROBLEM_COPY"
    : > "$TRACE_DIR/output.json"

    apptainer exec \
    --bind "$PROBLEM_COPY:/problem" \
    --bind "$SETTINGS_JSON:/root/.claude/settings.json" \
    --bind "$TRACE_DIR/output.json:/output.json" \
    --bind "$CONTAINER_SCRIPT:/guest.sh" \
    "$AGENT_SIF" /guest.sh /problem
done
