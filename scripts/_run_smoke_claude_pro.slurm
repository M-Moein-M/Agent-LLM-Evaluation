#!/bin/bash -l
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=eval-%x-%j.out
#
# Smoke test for the claude-pro agent (Claude Code on a personal subscription,
# api.anthropic.com — NOT the gateway). One problem, one round, serial: the
# point is to prove auth + outbound HTTPS + the trace pipeline from a COMPUTE
# node before committing to a full wave. Scale up via the kimi-style wave script
# once this comes back green.

set -euo pipefail

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

# Repo root = wherever this was submitted from. Hardcoding an absolute path
# broke the moment the workspace moved off $WORK (invisible from the test
# cluster); SLURM_SUBMIT_DIR follows the repo instead of pinning it.
cd "$SLURM_SUBMIT_DIR"

export CONTAINER_RUNTIME=apptainer
export EVAL_AGENT=claude-pro
export EVAL_MODELS="${EVAL_MODELS:-claude-opus-5}"
export EVAL_ROUNDS=1
export EVAL_SOLVE_PAR=1
export EVAL_PROBLEMS="RabbitCT-LolaBunny"
export DATA_ROOT="$PWD/data"
export AGENT_SIF="$PWD/agents/apptainer/claude-pro/agent.sif"

echo "=== smoke: agent=$EVAL_AGENT model=$EVAL_MODELS rounds=$EVAL_ROUNDS par=$EVAL_SOLVE_PAR ==="
bash scripts/run_eval.sh
