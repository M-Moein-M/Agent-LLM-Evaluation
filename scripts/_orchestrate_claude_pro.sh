#!/bin/bash
# Login-node orchestrator: run the 4 graded problems one at a time on the
# claude-pro agent (5 runs each), with a 2h gap between problems. Each problem
# is a separate GPU job submitted with `sbatch --wait` so the A40 is released
# during the 2h sleeps instead of sitting idle.
set -uo pipefail

cd $WORK/nhr-agent-evaluation-remote

PROBLEMS=(RabbitCT-LolaBunny RabbitCT-LolaOMP tiled-gemm stream-compact)
GAP=7200   # 2h between problems
first=1

for p in "${PROBLEMS[@]}"; do
  if [ "$first" -eq 0 ]; then
    echo "[orch] $(date '+%F %T') sleeping ${GAP}s (2h) before $p"
    sleep "$GAP"
  fi
  first=0
  echo "[orch] $(date '+%F %T') submitting $p (--wait, blocks until done)"
  sbatch --wait \
    --job-name="cpro-$p" \
    --export=ALL,EVAL_PROBLEMS="$p" \
    scripts/_run_one_claude_pro.slurm
  rc=$?
  echo "[orch] $(date '+%F %T') $p job finished rc=$rc"
done

echo "[orch] $(date '+%F %T') all 4 problems complete"
