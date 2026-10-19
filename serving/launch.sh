#!/bin/bash
# Resolve the configuration, write a ready-to-submit sbatch file and submit it.
#
#   launch.sh <single-node|multi-node> [<model> [-- extra sbatch args]]
#   launch.sh single-node            list the models of a mode
#   DRY_RUN=1 launch.sh multi-node Kimi-K2.7-Code-TP4xPP3
#
# Why a wrapper: #SBATCH lines cannot expand variables, so the header of
# <mode>/serve.sbatch holds ${VAR} placeholders. This script envsubsts only the #SBATCH
# lines into a temp file, drops lines whose value is empty, and submits that file.
#
# DRY_RUN=1 submits nothing and starts nothing. It runs the same job body with
# Slurm, srun, apptainer, rsync and mkdir replaced by printing, and shows the resolved
# variables, every launch command (sglang, router, litellm, monitoring) and the
# generated prometheus.yml. Extra output: DRY_RUN_VERBOSE=1 also prints the rendered
# LiteLLM config.
set -euo pipefail

SERVING_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SERVING_DIR
# shellcheck disable=SC1091
. "$SERVING_DIR/common/lib.sh"

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }

[ $# -ge 1 ] || usage
mode=$1
shift
case $mode in
  single-node | multi-node) ;;
  -h | --help) usage 0 ;;
  *) echo "unknown mode '$mode'" >&2; usage ;;
esac

load_env

models_dir=$SERVING_DIR/$mode/models
if [ $# -eq 0 ] || [ "$1" = -- ]; then
  echo "models for $mode:"
  for f in "$models_dir"/*.sh; do basename "$f" .sh; done | sed 's/^/  /'
  exit 0
fi
model=$1
shift
[ "${1:-}" = -- ] && shift
sbatch_args=("$@")
[ -f "$models_dir/$model.sh" ] || die "no $models_dir/$model.sh (run launch.sh $mode to list models)"

# Read what the header needs from the model file, in a subshell so nothing leaks.
model_info=$(
  # shellcheck disable=SC1090,SC1091
  . "$SERVING_DIR/common/lib.sh"
  . "$SERVING_DIR/single-node/lib.sh"
  MODEL_PATH=${MODEL_DIR%/}
  . "$models_dir/$model.sh"
  configure_model
  if [ "$mode" = single-node ]; then NODES=1; fi
  printf 'NODES=%s REQUIRED_GPUS=%s RECOMMENDED_CPUS=%s\n' "$NODES" "${REQUIRED_NUM_GPUS:-$((TP_SIZE * ${PP_SIZE:-1}))}" "${RECOMMENDED_CPUS_PER_TASK:-}"
)
eval "$model_info"
export NODES
if [ "$mode" = single-node ] && [ "$REQUIRED_GPUS" -gt "$GPUS_PER_NODE" ]; then
  echo "WARNING: $model needs $REQUIRED_GPUS GPUs, GPUS_PER_NODE=$GPUS_PER_NODE" >&2
fi
if [ "$mode" = single-node ]; then
  export JOB_NAME=${JOB_NAME:-sgl-serve}
else
  export JOB_NAME=${JOB_NAME:-sgl-serve-mn}
fi
export CPUS_PER_TASK=${CPUS_PER_TASK:-${RECOMMENDED_CPUS:-32}}
if [ -n "${SBATCH_GRES:-}" ]; then
  export SBATCH_GPU_FLAG="--gres=$SBATCH_GRES"
else
  export SBATCH_GPU_FLAG="--gpus-per-node=$GPUS_PER_NODE"
fi
export MODE=$mode MODEL=$model

header_vars='${JOB_NAME} ${SBATCH_PARTITION} ${SBATCH_ACCOUNT} ${NODES} ${SBATCH_GPU_FLAG} ${CPUS_PER_TASK} ${TIME_LIMIT} ${TIME_MIN} ${SBATCH_NODELIST} ${SBATCH_EXTRA} ${SERVE_ROOT} ${LOG_DIR}'

render_sbatch() {
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    if [[ $line == '#SBATCH'* ]]; then
      line=$(printf '%s' "$line" | envsubst "$header_vars")
      # drop options whose value is empty ("--time-min=", a bare "#SBATCH ")
      if [[ $line =~ ^#SBATCH[[:space:]]*$ ]] || [[ $line =~ =$ ]]; then continue; fi
    fi
    printf '%s\n' "$line"
  done <"$SERVING_DIR/$mode/serve.sbatch"
}

tmp=$(mktemp "${TMPDIR:-/tmp}/serve-${mode}-XXXXXX.sbatch")
trap 'rm -f "$tmp"' EXIT
render_sbatch >"$tmp"
# SBATCH_* are also input variables of sbatch itself (an empty SBATCH_GRES would be parsed as
# --gres=). The values are in the header now, do not hand them to sbatch a second time.
unset SBATCH_PARTITION SBATCH_ACCOUNT SBATCH_GRES SBATCH_NODELIST SBATCH_EXTRA SBATCH_GPU_FLAG

if is_dry; then
  echo "== DRY RUN: $mode / $model (nothing is submitted or started) =="
  echo "-- generated #SBATCH header --"
  grep '^#SBATCH' "$tmp" | sed 's/^/  /'
  echo "-- job body, executed locally with Slurm and container commands printed --"
  DRY_RUN=1 bash "$tmp"
  echo "== DRY RUN done =="
  exit 0
fi

mkdir -p "$LOG_DIR" "$RUN_DIR"
echo "submitting $tmp ($mode / $model)"
sbatch ${sbatch_args[@]+"${sbatch_args[@]}"} "$tmp"
