#!/bin/bash -l
#SBATCH --job-name=agent-eval
#SBATCH --gres=gpu:a40:1            # one GPU (matches your interactive salloc; swap a40->a100 if needed)
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32          # >= EVAL_SOLVE_PAR: each parallel solve runs node/python + CUDA compiles on CPU
#SBATCH --mem=64G                   # ADJUST: ~a few GB per concurrent solve
#SBATCH --output=eval-%j.out
#
# Run the agent-evaluation harness on one GPU via Apptainer.
#
#   sbatch scripts/run_eval.slurm
#   # or interactively:  salloc --gres=gpu:a40:1 --time=02:00:00  then bash this file's body
#
# Prereqs (do once):
#   * Build the agent SIFs with fakeroot:
#       APPTAINER_CACHEDIR=$TMPDIR/apptainer-cache \
#         apptainer build --fakeroot agents/apptainer/pi/agent.sif       agents/apptainer/pi/agent.def
#       APPTAINER_CACHEDIR=$TMPDIR/apptainer-cache \
#         apptainer build --fakeroot agents/apptainer/swe-mini/agent.sif agents/apptainer/swe-mini/agent.def
#   * Ensure the compute node can reach the sglang gateway in models.json (baseUrl).
#   * Ensure the bulk RabbitCT data is present under $DATA_ROOT (Stage 1 of
#     run_eval.sh will download it if missing — put it on persistent scratch,
#     NOT $TMPDIR, so it isn't re-downloaded every job).

set -euo pipefail

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true   # ADJUST to your cluster

cd "$SLURM_SUBMIT_DIR"   # run from the repo root you submitted from

# --- runtime + eval configuration (override at submit time with --export) ---
export CONTAINER_RUNTIME=apptainer
export EVAL_AGENT="${EVAL_AGENT:-pi}"
export EVAL_MODELS="${EVAL_MODELS:-Qwen3.6-35B-A3B}"
export EVAL_ROUNDS="${EVAL_ROUNDS:-10}"
export EVAL_SOLVE_PAR="${EVAL_SOLVE_PAR:-32}"
export EVAL_PROBLEMS="${EVAL_PROBLEMS:-RabbitCT-LolaBunny,RabbitCT-LolaOMP,tiled-gemm,stream-compact}"

# Per-agent SIF (host.sh defaults to agents/apptainer/<agent>/agent.sif).
export AGENT_SIF="${AGENT_SIF:-$PWD/agents/apptainer/$EVAL_AGENT/agent.sif}"

# Bulk input data on persistent storage (read-only bind); per-job workdirs land
# on node-local SSD automatically because host.sh uses mktemp, which honors
# $TMPDIR (set by SLURM to the node-local scratch).
export DATA_ROOT="${DATA_ROOT:-$PWD/data}"

bash scripts/run_eval.sh
