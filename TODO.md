
# TODO — Evaluation Harness

## Target: first clean A40 run (current push)

Goal state: launch **all 4 problems on A40**, models supplied via `EVAL_MODELS`
at submit time, with statistics (done) **+ a minimal per-problem perf baseline**.
Agents for this push: **mini-swe + claude** (pi dropped — no enforceable step
cap; see Agent expansion). Apptainer-only for now (docker parity is later).
Good-enough / minimalist; normalize and polish after the first run.

### Milestone A — make mini-swe + claude launchable

- [x] **Complete the claude apptainer agent.** DONE — added `host.sh` (mirrors
  `agents/apptainer/swe-mini/host.sh`) + `solve.sh` (runs Claude Code headless,
  Anthropic protocol → `agent.stream.jsonl`), reusing the agent-agnostic
  `agents/measure.sh` for Phase 2; rewrote `agent.def` onto the
  CUDA-devel base (+nsys) so the measure phase can compile/run. Cap: Claude Code
  `--max-turns` (`EVAL_MAX_TURNS`, default 120). Compaction: `trace_stats.py`
  gained a `parse_claude` for the stream-json trace (turns/steps/tokens +
  context-drop compaction detection). The old `generate.sh`/`guest.sh` one-shot
  stubs are superseded by `solve.sh`. **Endpoint resolved:**
  point Claude Code at the gateway by exporting before launch —
  `ANTHROPIC_BASE_URL=https://gateway.example.org/api/llmgw`, `ANTHROPIC_API_KEY=...`,
  and `ANTHROPIC_DEFAULT_{SONNET,OPUS,HAIKU}_MODEL=<gateway model id>` (so the
  gateway models map onto Claude Code's tier slots; the agent still talks the
  Anthropic protocol, just to our gateway). **Point all three tier slots
  (SONNET/OPUS/HAIKU) at the same gateway base URL and the same single model
  id.** The gateway serves one model at a time (see run strategy below), and
  Claude Code routes some internal calls to the HAIKU slot — if any slot points
  at a model that isn't currently served, claude silently fails on that tier. So
  `host.sh` should default all three `ANTHROPIC_DEFAULT_*_MODEL` to the single
  `EVAL_MODELS` entry (override only if all targeted models are served at once).
  **Verify it has an enforceable cap**
  (turns/cost) so it can't run away like pi; ensure context compaction is
  captured (meeting warning — `trace_stats.py` already records `compacted_runs`).
- [x] **Drop pi from the run.** DONE — default `EVAL_AGENT` switched to
  `swe-mini` in both `run_eval.slurm` and `run_eval.sh`; pi dir kept but not
  default. Run per-agent (one sbatch per `EVAL_AGENT`), e.g.
  `sbatch --export=ALL,EVAL_AGENT=claude scripts/run_eval.slurm`.
- [x] **Reachability preflight (Stage 0).** DONE — `run_eval.sh` now probes each
  id in `EVAL_MODELS` against the gateway `baseUrl` (1-token completion, 30 s
  budget) before any data prep/solve, aborting with a clear per-model error if
  any returns non-200. Skip with `EVAL_SKIP_PREFLIGHT=1`. Models come from
  `EVAL_MODELS` at submit time.

### Run strategy — per-model waves (gateway serves one model at a time)

The gateway can only serve a single model endpoint at a time, so the first
A40 run is organized as **per-model waves**, not one big multi-model launch:
serve model X → `sbatch` one job per agent with `EVAL_MODELS=X` → tear down →
serve model Y → repeat. This is fully supported: `EVAL_MODELS` is set at submit
time and `summary.json` aggregates by `agent → model`, so waves run at different
times still merge into one aggregate. Set `EVAL_MODELS` to the **single** served
model per wave (a one-element list); the reachability preflight (A3) then doubles
as the gate confirming the currently-served model is live before a 2 h job starts.

### Milestone B — scale sizes to A40

- [ ] **RabbitCT → `-s 1024`.** `EVAL_SIZE` 256→1024; download `Reference1024.vol`
  via the RabbitCT download script (confirmed available); update each problem's
  `PROMPT.md` (size text **and** example `solution.json`).
- [ ] **tiled-gemm + stream-compact → bigger.** Pick A40-scale sizes (proposal:
  gemm ~4096³, stream-compact ~1e8 elements — tune), update `PROMPT.md` size text
  + source + example `solution.json`.
- [ ] **Retune thresholds for the larger sizes.** GPU-util bars in
  `checks/*/check.py` were set for a saturated GTX 1050; revisit them, and bump
  `environment.timeout` (`solve.sh` / see AGENT.md) since larger sizes run longer.

### Milestone C — minimal per-problem baseline

- [ ] **Record kernel/execution time as an informational (non-gating) metric on
  all 4 problems.** gemm already has the cuBLAS ratio (keep it). For the other 3,
  capture the kernel/run time from the run (parse the run log or wrap with a
  timer), store it alongside tokens/steps, and surface it in
  `collect_results.py` / `summary.json`. **Do NOT make it pass/fail.** Normalize
  / add a roofline layer later. (Meeting items: compare-to-cuBLAS / Nvidia
  numbers.)

### Still open / loose ends (not blocking the first run)

- [ ] **Tune `PERF_MIN_FRAC`** (gemm, provisional 0.15) against real A40 numbers —
  a floor that rejects naive kernels but passes a genuine optimization.
- [ ] **ncu roofline / %-of-peak layer** — richer than the cuBLAS/time ratios, but
  needs GPU perf-counter access — **verify it's permitted on cluster/NHR**
  (`NVreg_RestrictProfilingToAdminUsers`) before wiring ncu in.
- [ ] **Granular "no-solution" failure view** — attribute each non-pass to a
  concrete cause (never called a tool / out of steps/tokens / wrong interface /
  stuck debug loop) from `agent.jsonl` + `.eval/facts.json`, instead of a bare
  `[FAIL]`. Post-run analysis; high value at 4-problem × N-round × multi-model
  scale. (Meeting's "why don't we have solutions?" item.)
- **SIF (re)builds are done by hand (owner: you)** — including the
  `nsight-systems-cli` image rebuild the gemm nsys timing needs (without it the
  gemm perf check falls back to the bench's cudaEvent timing).

## Agent expansion (after the first run)

- [ ] **swe (full SWE-agent).** Build the apptainer integration mirroring
  `swe-mini` (`agent.def` installs swe-agent; `solve.sh` / `measure.sh` /
  `host.sh`). Same family as mini → `step_limit`/`cost_limit` caps are available;
  set one.
- **Aider — SET ASIDE (decision).** Aider does **not** give a tunable step limit,
  and it doesn't fit a "set `max_steps = N`" uniform knob across tiers. Its only
  guard is the **fixed 3-reflection ceiling** (not configurable). If the benchmark
  design relies on every agent exposing a configurable iteration cap, Aider is the
  odd one out — it fails that the same way **pi failed the limiting-config check**.
  It's also why Aider sits at **Tier 1**: it's the least agentic candidate, closer
  to a one-shot editor than an autonomous loop. If a configurable cap is genuinely
  a must, we'd want a different Tier-1 pick — but, excluding pi, there isn't an
  obvious Tier-1 candidate with a clean step knob at that capability level.
  Parking Aider for now rather than building it.

## Docker parity (later — deferred from this push)

- [ ] Port the new apptainer agents (claude, plus swe once built) back to
  `agents/docker/` for local runs — `agents/docker/` currently has only pi +
  swe-mini. Keep the two-phase contract identical so checks/results are shared.

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
