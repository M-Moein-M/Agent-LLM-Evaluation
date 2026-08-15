#!/bin/bash -l
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=eval-%x-%j.out
#
# One agent wave over the 4 graded problems x 10 rounds against the
# moonshotai/Kimi-K2.7-Code endpoint. Per-agent vars (EVAL_AGENT, EVAL_MODELS)
# come in via `sbatch --export`; the shared vars (incl. the comma-bearing
# EVAL_PROBLEMS, which cannot go through --export) are set here.

set -euo pipefail

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

# Repo root = wherever this was submitted from. Hardcoding an absolute path
# broke the moment the workspace moved off $WORK (invisible from the test
# cluster); SLURM_SUBMIT_DIR follows the repo instead of pinning it.
cd "$SLURM_SUBMIT_DIR"

export CONTAINER_RUNTIME=apptainer
export EVAL_PROVIDER=local
export EVAL_ROUNDS=10
export EVAL_SOLVE_PAR=4
export EVAL_PROBLEMS="RabbitCT-LolaBunny,RabbitCT-LolaOMP,tiled-gemm,stream-compact"
export DATA_ROOT="$PWD/data"
export AGENT_SIF="$PWD/agents/apptainer/$EVAL_AGENT/agent.sif"

echo "=== wave: agent=$EVAL_AGENT model=$EVAL_MODELS rounds=$EVAL_ROUNDS par=$EVAL_SOLVE_PAR ==="
echo "=== problems: $EVAL_PROBLEMS ==="
bash scripts/run_eval.sh
