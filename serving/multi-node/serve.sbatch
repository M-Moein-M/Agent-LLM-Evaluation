#!/bin/bash -l
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --nodes=${NODES}
#SBATCH ${SBATCH_GPU_FLAG}
#SBATCH --ntasks=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --time-min=${TIME_MIN}
#SBATCH --nodelist=${SBATCH_NODELIST}
#SBATCH ${SBATCH_EXTRA}
# Nodes may not mount the submit directory or your home. Pin the working dir and stdout
# onto the shared workspace (SERVE_ROOT) or Slurm can fail the step before the script runs.
#SBATCH --chdir=${SERVE_ROOT}
#SBATCH --output=${LOG_DIR}/%x-%j.log

# Multi-node serving: one task per node, each running one SGLang process that owns the
# node's GPUs, together one TP x PP group. NODES comes from the model
# (TP_SIZE * PP_SIZE / GPUS_PER_NODE). LiteLLM and the monitoring stack run on node 0.
# The model comes from multi-node/models/$MODEL.sh.
#
# The #SBATCH lines above are templates, see single-node/serve.sbatch. Use launch.sh.

set -euo pipefail
unset SLURM_EXPORT_ENV

: "${SERVING_DIR:?run through serving/launch.sh}"
: "${MODEL:?run through serving/launch.sh}"
# shellcheck disable=SC1091
{
  . "$SERVING_DIR/common/lib.sh"
  . "$SERVING_DIR/common/monitoring.sh"
  . "$SERVING_DIR/common/litellm.sh"
  . "$SERVING_DIR/multi-node/lib.sh"
}
load_env

MODEL_PATH=${MODEL_DIR%/}
# shellcheck disable=SC1090
. "$SERVING_DIR/multi-node/models/$MODEL.sh"
configure_model
if is_dry; then dry_slurm_env "$NODES"; fi

log "multi-node: model=$MODEL job=${SLURM_JOB_ID:-?} node(s)=$(list_nodes | tr '\n' ' ')"
print_vars MODEL_TAG MODEL_NAME TP_SIZE PP_SIZE NODES GPUS_PER_NODE CTX_SIZE HTTP_PORT DIST_PORT SGLANG_SIF_PATH \
  MODEL_DIR IMAGE_DIR RUN_DIR MEM_FRACTION REASONING_PARSER TOOL_CALL_PARSER LITELLM_PORT LITELLM_BIN
if ! is_dry; then ip addr show | grep inet || true; fi

# Engine /metrics lives on node 0's HTTP port. Deliberately not named SGLANG_PORT
# (see the port note in multi-node/models/Kimi-K2.7-Code-TP4xPP3.sh).
SGLANG_METRICS_PORTS=$HTTP_PORT
# dcgm-exporter on every node + prometheus/grafana on node 0, started before sglang so
# metrics flow during the weight load. Torn down with the job.
start_monitoring

# litellm front door in the background on node 0 (it tolerates sglang not being up yet --
# calls just fail until the model finishes loading). Dies with the job.
LITELLM_PID=0
if is_dry; then
  start_litellm "$HTTP_PORT"
else
  start_litellm "$HTTP_PORT" &
  LITELLM_PID=$!
fi
cleanup() {
  if is_dry; then return 0; fi
  kill "${LITELLM_PID:-0}" 2>/dev/null || true
  stop_monitoring
}
trap cleanup EXIT

# serve_sglang is the job's main process and srun-fans-out across all nodes.
serve_sglang
