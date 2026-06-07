# TODO — Evaluation Harness

_No open items._

---

## Previous decisions (for reference, not action items)

- **Agent writes code + `solution.json`, harness replays commands** — gives the
  agent debugging agency (fix in one pass), harness independently re-runs the
  build/run to verify.
- **One output format for all problems** — `solution.json` works across RabbitCT,
  MD-Bench, etc.; only the prompt changes per problem.
- **Host-side pass judgment** — `guest.sh` captures raw evidence only
  (`.eval/facts.json` + logs + `gpu_usage.csv`); pass/fail is decided on the host
  in `checks/` (`eval_lib.py` engine + per-problem `check.py` thresholds →
  `pass_result.json`, aggregated by `collect_results.py`). Re-runnable over
  existing traces with no GPU cost; lives outside the agent workspace so it can't
  be gamed.
- **`agent.jsonl` trace** — full event stream, but the high-frequency
  `message_update` / `tool_execution_update` deltas are filtered out before
  writing (they re-serialize the whole message per token → quadratic, ~99.7% of
  the bytes). Final content is preserved losslessly in `message_end` /
  `agent_end`.
- **GPU sampling** — single background `nvidia-smi --format=csv,nounits -lms 250`
  writer for the duration of the run phase, killed afterward; utilization
  aggregated in `checks/eval_lib.py:gpu_stats` and thresholded per problem.
- **Bulk input data externalized** — large per-problem inputs (e.g. RabbitCT's
  ~4.5 GB `RabbitInput/`) live in the gitignored repo-root `data/` dir and are
  bind-mounted read-only into the container, never copied per round. `run_eval.sh`
  Stage 1 downloads them if missing.
