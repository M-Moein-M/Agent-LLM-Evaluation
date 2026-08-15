#!/bin/bash -l
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=eval-%x-%j.out
#
# One graded problem, claude-pro agent (Claude Code on a personal subscription,
# api.anthropic.com — NOT the gateway), 5 serial runs. The problem comes in
# via `sbatch --export=ALL,EVAL_PROBLEMS=<name>`. The orchestrator submits these
# one at a time with a 2h gap between problems (subscription rate-limit spacing).

set -euo pipefail

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

# Repo root = wherever this was submitted from. Hardcoding an absolute path
# broke the moment the workspace moved off $WORK (invisible from the test
# cluster); SLURM_SUBMIT_DIR follows the repo instead of pinning it.
cd "$SLURM_SUBMIT_DIR"

export CONTAINER_RUNTIME=apptainer
export EVAL_AGENT=claude-pro
export EVAL_MODELS="${EVAL_MODELS:-claude-opus-5}"
export EVAL_ROUNDS=5
export EVAL_SOLVE_PAR=1
export EVAL_PROBLEMS="${EVAL_PROBLEMS:?set EVAL_PROBLEMS via --export}"
export DATA_ROOT="$PWD/data"
export AGENT_SIF="$PWD/agents/apptainer/claude-pro/agent.sif"

echo "=== one: agent=$EVAL_AGENT model=$EVAL_MODELS problem=$EVAL_PROBLEMS rounds=$EVAL_ROUNDS par=$EVAL_SOLVE_PAR ==="
bash scripts/run_eval.sh
