#! /bin/bash -l

PROBLEM_DIR=/home/${GROUP}/${USER}/agent-evaluation-nhr/problems/
AGENT_SIF=/home/${GROUP}/${USER}/agent-evaluation-nhr/agents/pi/agent.sif
CONTAINER_SCRIPT=/home/${GROUP}/${USER}/agent-evaluation-nhr/agents/pi/guest.sh

# agent specific config
MODELS_CONFIG=/home/${GROUP}/${USER}/agent-evaluation-nhr/agents/pi/models.json
EVAL_TMPDIR=$(mktemp -d)
cp -r $PROBLEM_DIR/* $EVAL_TMPDIR/
for PROBLEM in $EVAL_TMPDIR/*; do
    echo "Evaluating problem: $PROBLEM"
    mkdir -p $PROBLEM/.pi/agent/
    cp $MODELS_CONFIG $PROBLEM/.pi/agent/
    apptainer exec \
    --bind $PROBLEM:/problem \
    --bind $MODELS_CONFIG:/models.json \
    --bind $CONTAINER_SCRIPT:/guest.sh \
    $AGENT_SIF /guest.sh /problem
done