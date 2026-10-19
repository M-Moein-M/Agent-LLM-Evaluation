#!/bin/bash -l
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH ${SBATCH_GPU_FLAG}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --time-min=${TIME_MIN}
#SBATCH --nodelist=${SBATCH_NODELIST}
#SBATCH ${SBATCH_EXTRA}
# Nodes may not mount the submit directory or your home. Pin the working dir and stdout
# onto the shared workspace (SERVE_ROOT) or Slurm can fail the step before the script runs.
#SBATCH --chdir=${SERVE_ROOT}
#SBATCH --output=${LOG_DIR}/%x-%j.log

# Single-node serving: one sglang process (or a router fleet) on this node's GPUs,
# LiteLLM on :$LITELLM_PORT in front, monitoring beside it. The model comes from
# single-node/models/$MODEL.sh.
#
# The #SBATCH lines above are templates: Slurm cannot expand variables in them, so
# launch.sh envsubsts them into a temp copy and submits that. Lines with an empty
# value are dropped. Submitting this file directly does not work, use launch.sh.
# DRY_RUN=1 runs this same body with Slurm commands printed instead of executed.

set -euo pipefail
unset SLURM_EXPORT_ENV

: "${SERVING_DIR:?run through serving/launch.sh}"
: "${MODEL:?run through serving/launch.sh}"
# shellcheck disable=SC1091
{
  . "$SERVING_DIR/common/lib.sh"
  . "$SERVING_DIR/common/monitoring.sh"
  . "$SERVING_DIR/common/litellm.sh"
  . "$SERVING_DIR/single-node/lib.sh"
}
load_env
if is_dry; then dry_slurm_env 1; fi

MODEL_PATH=${MODEL_DIR%/}
# shellcheck disable=SC1090
. "$SERVING_DIR/single-node/models/$MODEL.sh"
configure_model
# Plain `sglang serve`: engine metrics on the serving port. declare_sglang_fleet already set it for fleets.
SGLANG_METRICS_PORTS=${SGLANG_METRICS_PORTS:-$SGLANG_PORT}

log "single-node: model=$MODEL job=${SLURM_JOB_ID:-?} node(s)=$(list_nodes | tr '\n' ' ')"
print_vars MODEL_TAG MODEL_NAME REQUIRED_NUM_GPUS CTX_SIZE SGLANG_PORT SGLANG_SIF_PATH MODEL_DIR STAGE_DIR \
  STAGE_MODEL IMAGE_DIR RUN_DIR LITELLM_PORT LITELLM_BIN SGLANG_METRICS_PORTS SGLANG_ROUTER_METRICS_PORT \
  NUM_WORKERS FLEET_WORKERS SGLANG_WORKER_TP
if ! is_dry; then ip addr show | grep inet || true; fi

# dcgm-exporter + prometheus + grafana, sized to this run's actual node(s) --
# started before sglang so metrics are already flowing once load/serving begins.
start_monitoring

# Start the server
stage_model
# litellm front door in the background (it tolerates sglang not being up yet --
# calls just fail until the model finishes loading); sglang stays in the foreground
# as the job's main process. Everything (litellm + monitoring stack) dies together
# when the job ends.
LITELLM_PID=0
if is_dry; then
  start_litellm "$SGLANG_PORT"
else
  start_litellm "$SGLANG_PORT" &
  LITELLM_PID=$!
fi
cleanup() {
  # SGLANG_WORKER_PIDS is set only by run_sglang_router_fleet (background workers
  # behind the router); killing them here keeps stop_monitoring running on the way out.
  if is_dry; then return 0; fi
  kill "${LITELLM_PID:-0}" ${SGLANG_WORKER_PIDS[@]+"${SGLANG_WORKER_PIDS[@]}"} 2>/dev/null || true
  stop_monitoring
}
trap cleanup EXIT
serve_sglang
