# TODO — Evaluation Harness

- **pi agent has no iteration cap → runaway "long tail" solves.** Observed on the
  4-problem PAR=16 local run: swe-mini (which has `step_limit: 80`) drained
  cleanly, but two pi solves ran 2 h+ in endless debug loops and never
  terminated, blocking Phase 1 from completing (the orchestrator waits on all
  solves). pi's CLI has no step/iteration/turn flag (confirmed via `pi --help`).
  Need a way to bound a pi solve. Candidate solutions to evaluate later:
  - **Token-budget limit** — cap total tokens spent per solve. Endpoint-speed
    *independent* (bounds the actual work/cost regardless of how slow the gateway
    is) → preferred direction. pi exposes no native flag, so needs a mechanism
    (e.g. track cumulative usage from the event stream and abort, or a
    gateway-side per-request budget).
  - **Wall-clock timeout** — simplest (wrap the solve call), but a *slow endpoint
    inflates wall-clock*, so it would kill legitimately-progressing agents. Poor
    fit on its own; only acceptable as a coarse last-resort backstop.

- **Apply the pending size + step-limit batch (before the cluster run).** Held
  back so it wouldn't disrupt the in-flight local runs; not yet applied.
  - **Increase problem sizes** — kernels must run long enough to measure on fast
    GPUs (A40/A100) and to actually fill them. RabbitCT (LolaBunny **and**
    LolaOMP): `-s 256 → -s 1024` and `Reference256.vol → Reference1024.vol`;
    tiled-gemm + stream-compact → larger matrix/array sizes. Edit each problem's
    `PROMPT.md` (the size text **and** the example `solution.json`), bump the
    `EVAL_SIZE` default, and revisit the GPU-util thresholds in
    `checks/*/check.py` (they were tuned for a saturated GTX 1050).
  - **swe-mini `step_limit: 80 → 150`** in `agents/docker/swe-mini/mini.yaml` —
    its RabbitCT failures were step-budget exhaustion ("no solution produced"),
    not bad code, so more steps should directly help.

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
