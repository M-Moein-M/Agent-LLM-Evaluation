# AGENT.md — operational guide for AI agents working in this repo

This file is for **you, an AI agent editing this repository**. It covers the
repo layout, how a run works end-to-end, how to run/test a single problem, what
to never touch, where output goes, and the environment gotchas that will waste
your time if you rediscover them by trial and error.

For the human-facing project overview, motivation, and architecture, see
[`README.md`](README.md). Don't duplicate that content here.

---

## TL;DR run command

The pipeline is Linux + Docker and **must run from WSL (Ubuntu-22.04)** on this
machine — not PowerShell, not Git-bash/msys (see [Gotchas](#gotchas)):

```bash
wsl -d Ubuntu-22.04 -- bash -lc 'cd /mnt/d/projects/nhr-agent-evaluation && \
  EVAL_MODELS=Qwen3.6-35B-A3B EVAL_ROUNDS=2 bash scripts/run_eval.sh'
```

When invoking `wsl` from a non-WSL shell, prefix with `MSYS_NO_PATHCONV=1` so the
outer shell doesn't rewrite the command string.

### swe-mini full run — all 4 graded problems (reference invocation)

The standard swe-mini benchmark over every problem that has a check, 10 rounds
each, against the Qwen3.6 model (model id `Qwen/Qwen3.6-35B-A3B-FP8` when served
via the NHR endpoint — it differs from `Qwen3.6-35B-A3B` depending on how the
model is served):

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- bash -lc 'cd /mnt/d/projects/nhr-agent-evaluation && \
  EVAL_AGENT=swe-mini \
  EVAL_MODELS="Qwen/Qwen3.6-35B-A3B-FP8" \
  EVAL_ROUNDS=10 \
  EVAL_PROBLEMS="RabbitCT-LolaBunny,RabbitCT-LolaOMP,tiled-gemm,stream-compact" \
  EVAL_SOLVE_PAR=10 \
  bash scripts/run_eval.sh'
```

40 jobs (4 problems × 10 rounds). `EVAL_SOLVE_PAR=10` runs all solves at once
(shared GPU; solve is gateway-bound). The mini agent's per-task iteration cap is
`agent.step_limit` in `agents/docker/swe-mini/mini.yaml` (currently **120**).

> **When you scale the problem size, raise the per-command timeout too.**
> `environment.timeout` in `agents/docker/swe-mini/solve.sh` (currently **600s**)
> is a *per-command* cap. At the current **256³** RabbitCT scale a correct
> build+run finishes well under it, so the timeouts you see are the agent's own
> pathological kernels hanging — leave 600s, don't raise it (raising it just
> lengthens the wall-clock tail on bad commands without helping correct ones).
> But if you bump RabbitCT to **1024³** (the pending cluster change) or use much
> larger GEMM / stream-compact sizes, a *legitimate* build+run can exceed 600s
> and get clipped — increase `environment.timeout` proportionally to the new
> scale at the same time you raise `EVAL_SIZE`. Same logic may apply to
> `step_limit` if bigger problems need more iterations to converge.

---

## Repo layout

```
scripts/run_eval.sh          Single entrypoint. Sets EVAL_* defaults, does Stage-1
                             data prep, then execs agents/docker/$EVAL_AGENT/host.sh.

agents/
  docker/                    CURRENT focus (easier GPU access).
    pi/                      pi agent (two-phase, parallel-capable):
      Dockerfile               CUDA base + Node + pi-coding-agent.
      host.sh                  Orchestrator: build image, build job list,
                               Phase 1 (solve, parallel) → Phase 2 (measure, serial),
                               extract traces, run host-side checks.
      solve.sh                 In-container Phase 1: run agent → source +
                               solution.json + agent.jsonl. NO measurement.
      measure.sh               In-container Phase 2: replay build/run with GPU
                               sampling → .eval/ evidence + facts.json. NO judgment.
                               Plus an OPTIONAL perf-grading block: if
                               /grade/bench.cu is bound (problem has one), compile
                               it against the agent kernel + cuBLAS and time both
                               (nsys, else cudaEvent) → .eval/perf.log + kern_*.csv.
    swe-mini/                swe-mini agent (two-phase, mirrors pi):
      Dockerfile               CUDA base + Python + mini-swe-agent (litellm).
      host.sh                  Orchestrator (mirror of pi's).
      solve.sh                 In-container Phase 1: run mini agent → source +
                               solution.json + agent.traj.json. NO measurement.
      measure.sh               In-container Phase 2: replay build/run with GPU
                               sampling → .eval/ evidence + facts.json. NO judgment.
      mini.yaml                Prompt templates + finite step_limit.
  apptainer/                 CLUSTER runtime (SLURM / cluster A40), selected by
                             CONTAINER_RUNTIME=apptainer. Same two-phase contract
                             as docker/. Agents: pi, swe-mini, claude, kimi.
      claude/                Claude Code agent (apptainer-only — no docker twin yet):
        agent.def              CUDA devel base + Claude Code native install + nsys.
        host.sh                Orchestrator (mirror of swe-mini's apptainer host.sh).
        solve.sh               Phase 1: run Claude Code headless against the NHR
                               gateway (Anthropic protocol) → source +
                               solution.json + agent.stream.jsonl. --max-turns cap.
                               Reuses docker/swe-mini/measure.sh for Phase 2.
      kimi/                  Kimi Code agent (apptainer-only — no docker twin yet):
        agent.def              CUDA devel base + Kimi Code native install
                               (KIMI_INSTALL_DIR=/usr/local → /usr/local/bin/kimi) + nsys.
        host.sh                Orchestrator (mirror of claude's apptainer host.sh).
        solve.sh               Phase 1: run Kimi Code headless against the NHR
                               gateway (OpenAI-compat — /v1 kept) → source +
                               solution.json + agent.stream.jsonl. Cap via config.toml
                               loop_control.max_steps_per_turn. Reuses
                               docker/swe-mini/measure.sh for Phase 2.
                             NOTE: models.json default still lives in apptainer/pi/.
      claude-pro/            Claude Code on a PERSONAL Anthropic subscription
                             (OAuth to api.anthropic.com — NOT the gateway,
                             never an open-weight model). The study's only
                             frontier-model row (claude-opus-4-8).
        agent.def              Same base as claude/, but auth is the subscription
                               token, not a gateway key.
        host.sh / solve.sh     Phase 1 mirrors claude/; Phase 2 reuses
                               docker/swe-mini/measure.sh. Scratch/secret dirs
                               (.secrets/, .setup-home/, .cache/, .work/) are
                               gitignored.
        README.md              Token setup (claude setup-token) + run notes.
                             Driven by scripts/_run_one_claude_pro.slurm and
                             scripts/_orchestrate_claude_pro.sh (one problem at a
                             time, 5 rounds/problem, spaced for the session limit).

problems/<P>/                One directory per problem.
  PROMPT.md                  Task description + the solution.json contract the
                             agent must produce. Fed to the agent in solve.sh.
  (sources, Makefile, mk/, run-bench.sh, ...)   per-problem build/run assets.

checks/                      Agent-AGNOSTIC judging subsystem (outside agent workspace
                             so it can't be gamed). Host-side, python3 only.
  eval_lib.py                ProblemCheck base + evidence accessors:
                             facts(), log_value(pattern), gpu_stats().
                             Shared stage helpers: gpu_stage() (gates on PEAK
                             util only — see below) and metric_stage() (non-gating
                             ranking measurement).
  <P>/check.py               Per-problem pass criteria: ordered stages + threshold
                             constants at top of file. Each ends with a non-gating
                             metric() stage that records the problem's rankable
                             scalar (mse, or tiled-gemm's cublas_frac).
  report_tables.py           Emits the per-problem ranking tables (markdown): one
                             table per problem, agent × model, pass@1 beside the
                             continuous metric (mean ± sd over every measured
                             round). EXCLUDE_CAMPAIGNS drops harness artifacts.
  <P>/bench.cu               OPTIONAL harness-owned perf bench (tiled-gemm only so
                             far). Bound read-only into measure.sh as /grade/bench.cu;
                             never enters the agent workspace. See "Performance
                             grading" below.
  check_pass.py              Driver: runs each problem's check over a trace →
                             .eval/pass_result.json.
  collect_results.py         Aggregates verdicts → pass@1 per (agent, model), plus
                             per_problem_rankings() (the ranking rows report_tables
                             renders). normalize_model() strips serving prefixes so
                             one model is one row; rounds are attributed to a device
                             at the CAMPAIGN level so failing rounds (no gpu_info)
                             still count on the A40.

data/<P>/                    Bulk per-problem input (GITIGNORED). Bind-mounted
                             read-only into the container, never copied per round.

traces/<problem>/<agent>/<model>/<timestamp>/round-<N>/   Output. See below.
results/                     Aggregated output: runs.jsonl + summary.json.
```

### Problems present

Graded (have a matching `checks/<P>/check.py`): **RabbitCT-LolaBunny** (CUDA GPU
backprojection, the default), **RabbitCT-LolaOMP** (OpenMP variant),
**tiled-gemm**, **stream-compact** (CUDA). Experimental / no checks yet:
`2D-seq`, `MD-Bench`, `0`. The RabbitCT variants share one ~4.5 GB
`RabbitInput/` dataset.

---

## How a run works end-to-end

`scripts/run_eval.sh` is staged and selects the agent via `EVAL_AGENT`
(default `swe-mini`):

0. **Stage 0 (reachability preflight).** Probes every id in `EVAL_MODELS`
   against the gateway `baseUrl` (1-token completion, ~30 s budget) and aborts
   early with a clear error if any is unreachable — so a mis-served model fails
   in seconds, not hours into a batch. It also doubles as the per-wave gate that
   the currently-served model is live (the gateway serves one model at a time).
   Skip with `EVAL_SKIP_PREFLIGHT=1`. **claude-pro** takes a different branch: it
   does not use the gateway, so it probes the subscription token file exists and
   that the node can reach `api.anthropic.com` instead of doing a gateway
   completion (a real Anthropic id would always fail that). `run_eval.sh` also
   defaults `EVAL_MODELS=claude-opus-4-8` for this agent (a real Anthropic id,
   not a `models.json` gateway id).

1. **Stage 1 (data prep).** Ensures each RabbitCT variant's bulk input is
   present. One canonical copy lives in `data/RabbitCT/`; each requested variant
   gets a symlink `data/<variant> → RabbitCT` so the per-problem lookup
   (`$DATA_ROOT/<problem>/RabbitInput`) resolves to it — one download, no
   duplication. Downloads via `problems/RabbitCT-LolaBunny/download-input.sh` if
   missing/incomplete (< 4096 MB).

2. **Stage 2.** Execs `agents/docker/$EVAL_AGENT/host.sh`, which:
   - builds the agent image if the tag is missing,
   - builds the job list = models × rounds × problems (missing problems skipped),
   - gives each job its **own persistent workdir** (a copy of the problem dir),
   - runs the agent and replays its commands,
   - extracts the trace, and
   - runs `checks/check_pass.py` over `traces/` at the end.

### pi: two phases with opposite GPU needs

- **Phase 1 — SOLVE (parallel, shared GPU).** Up to `EVAL_SOLVE_PAR` (default 4)
  `solve.sh` containers run at once. The agent's own compile/run iterations only
  need *correctness* feedback and mostly wait on the LLM gateway, so GPU
  contention is fine. Each writes source + `solution.json` + `agent.jsonl` to its
  workdir.
- **Phase 2 — MEASURE (serial, GPU-isolated).** `measure.sh` replays each
  solution's build/run with 250 ms GPU sampling, producing the graded evidence.
  Strictly serial so each measured run owns the GPU — this is what timing/GPU
  thresholds are judged on. Set `EVAL_SOLVE_PAR=1` for fully-serial behavior.

### swe-mini: same two-phase split

swe-mini now mirrors pi: `solve.sh` (Phase 1, runs the mini agent → source +
solution.json + agent.traj.json, parallelizable up to `EVAL_SOLVE_PAR`) and
`measure.sh` (Phase 2, replay build/run with GPU sampling, strictly serial).
`measure.sh` is agent-agnostic — byte-for-byte the same logic as pi's.

### claude: token-budget & thinking decisions

Claude Code talks the Anthropic protocol, where **`max_tokens` is a required
field on every `/v1/messages` request** — unlike swe-mini's OpenAI-compat path,
which can omit it and let the server's own ceiling govern. So Claude Code always
RESERVES a fixed output slice against the model's single shared window
(`max_model_len = 262144 = input + output`). You cannot disable this; unset just
defaults Claude Code to 32000, and any value set is clamped to ≤128000.

**Output cap — `CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000` (the `EVAL_MAX_OUTPUT_TOKENS`
default).** The reservation trades directly against input headroom
(`headroom ≈ 262144 − cap`). Measured failure modes on Qwen3.6 + tiled-gemm:
- 32K (default): a verbose `thinking` block burns the whole budget and the turn
  stops with `stop_reason=max_tokens` BEFORE writing `solution.json` (truncated).
- 128K (=150K clamped): once conversation input grows past ~134K, `input+cap`
  exceeds 262144 and every call 400s with `ContextWindowExceededError` — fatal,
  since auto-compaction is disabled (no recovery).
- 64K: holds Qwen's largest thinking turns (~31K) plus the answer, and leaves
  ~186K input headroom — tiled-gemm reached ~119K input over ~30 turns with no
  400. This is the chosen value.

**Thinking — NOT controllable from the Claude Code side for this model.** Qwen3.6
is a reasoning model that free-runs native chain-of-thought (observed 12K–34K-token
thinking blocks per turn), which the gateway surfaces as Anthropic `thinking`
content blocks. `MAX_THINKING_TOKENS` (→ `thinking.budget_tokens`) is **a no-op
here**: Claude Code only emits a thinking budget for models it recognizes as
extended-thinking-capable, and our custom-named Qwen is not in its registry, so it
sends no thinking field at all. Verified empirically — forwarding
`EVAL_MAX_THINKING_TOKENS=4096` still produced 24K-token thinking blocks (3 gemm
runs all died at ~166–176K context with `"Prompt is too long"`). A raw-request
probe *does* show the API-layer lever exists (litellm maps `budget_tokens`: ≲1500
→ `'minimal'`, which this vLLM build **400s** on; ≈2048 → `low`; ≈4096 → `medium`;
≥8192 → `high`), but Claude Code never pulls it for this model. The only lever that
actually bounds Qwen thinking is Qwen-native (`/no_think` /
`chat_template_kwargs.enable_thinking`), which Claude Code cannot inject and which
would disable thinking entirely + alter the task prompt (fairness concern). So in
practice thinking is **always max** for claude+Qwen, which is exactly why the 64K
output cap (≥ the ~34K thinking peak) is required and cannot be lowered. The
`EVAL_MAX_THINKING_TOKENS` knob stays plumbed for a future model Claude Code *does*
recognize, but has no effect on the current served model.

**Why no compaction safety net.** `DISABLE_AUTO_COMPACT=1` for cross-agent
fairness (pi/swe-mini have no compaction), so a run that exhausts 262144 fails on
context exhaustion — the analog of the others running out of steps — rather than
silently summarizing to continue. The modest 64K cap keeps that wall from being
hit prematurely by an oversized reservation.

### kimi: openai-compat, env-config, config.toml step-cap

Kimi Code (Moonshot's terminal agent) is the simplest of the harnesses to wire up,
because it speaks the **OpenAI-compatible protocol natively** — none of claude's
Anthropic-protocol baggage applies:

- **No /v1 stripping** — the gateway `baseUrl` is used as-is (`/v1` included).
- **You MUST cap the output budget (`EVAL_MAX_OUTPUT_TOKENS`, default 64000).** Kimi
  derives `max_completion_tokens` as `min(context, 131072)` and so requests **131072**
  on every call by default. That is rejected *before any work happens*: the NHR
  gateway 429s ("tokens" rate limit, 100k/min) and Qwen3.6-27B 400s ("max 128000
  completion"). `solve.sh` exports `KIMI_MODEL_MAX_COMPLETION_TOKENS=$EVAL_MAX_OUTPUT_TOKENS`
  (read provider-agnostically in `resolveCompletionBudget`, ahead of the context-based
  fallback). NB: setting `KIMI_MODEL_MAX_CONTEXT_SIZE` alone does NOT help — with no
  hard cap the request is still `min(context, 131072) = 131072`. 64000 mirrors claude's
  working value on this gateway. So claude's `CLAUDE_CODE_MAX_OUTPUT_TOKENS` saga is
  NOT moot — kimi has the same constraint, just expressed through a different knob.
- **Model config via the `KIMI_MODEL_*` env channel** (set in `solve.sh`):
  `KIMI_MODEL_NAME` (=`EVAL_MODEL`), `KIMI_MODEL_API_KEY`, `KIMI_MODEL_PROVIDER_TYPE=openai`,
  `KIMI_MODEL_BASE_URL`. When `KIMI_MODEL_NAME` is set Kimi Code synthesizes an
  in-memory provider + model alias and makes it the `default_model`, so there's no
  provider block in `config.toml` and no `-m` flag. (Ordinary `OPENAI_API_KEY`-style
  vars are deliberately *not* read from the shell — this family is the one exception.)
- **Auto-approval is automatic in `-p` mode.** Print mode installs an approval
  handler that approves every tool call (the analog of claude's `bypassPermissions`),
  so there is no `--allowedTools` / `--permission-mode` flag and no `[permission]`
  block to write. Don't pass `--yolo`/`--auto`: print mode *rejects* them.
- **The enforceable cap lives in `config.toml`,** not a flag: `solve.sh` writes
  `[loop_control] max_steps_per_turn = $EVAL_MAX_TURNS`. In `-p` mode the run is a
  single turn, so this bounds the whole agentic loop — parity with swe-mini's
  `step_limit` and claude's `--max-turns`. Exceeding it raises
  `LOOP_MAX_STEPS_EXCEEDED` (the analog of running out of steps).
- **No compaction caveat needed.** Unlike claude (where `DISABLE_AUTO_COMPACT=1` is
  set for fairness), Kimi's `micro_compaction` is on by default; if cross-agent
  fairness later demands it be disabled, set
  `KIMI_CODE_EXPERIMENTAL_MICRO_COMPACTION=false`.
- **Install / HOME.** The native installer goes to `/usr/local/bin/kimi`
  (`KIMI_INSTALL_DIR=/usr/local`), so — unlike claude's `/root/.local` binary — the
  per-job staging `--home` can't shadow it. Each job still gets its own HOME so the
  `~/.kimi-code` data root (config.toml, sessions, logs) is isolated across parallel
  solves. `KIMI_INSTALL_DIR` (binary) and `KIMI_CODE_HOME` (data root, defaults to
  `$HOME/.kimi-code`) are distinct.

**Smoke status (2026-06-29, tiled-gemm + Qwen3.6-35B on gateway).** The harness
mechanically WORKS end-to-end: kimi connects under the `openai` provider type,
auto-approves tools in `-p` mode, Reads sources, runs `make`/`nvcc`, runs the GPU
binary, and writes a 14 KB `agent.stream.jsonl`. It still ends `[FAIL] (build)`
because it hit the **same Qwen runaway-thinking wall that banks tiled-gemm as a FAIL
for claude+Qwen** (see [[gemm-claude-context-wall]] in memory): Qwen free-ran native
chain-of-thought that exhausted the 64000 output budget in a single turn and returned
thinking-only content (`finishReason=truncated, rawFinishReason=length`), which kimi
treats as a fatal `provider.api_error` — so it never wrote `solution.json`. This is a
model/endpoint constraint, not a harness bug. Levers (untried / uncertain): raise
`EVAL_MAX_OUTPUT_TOKENS` toward ~96000 (must stay < the 100k/min TPM); or kimi's
thinking knobs `KIMI_MODEL_THINKING_MODE=off` / `KIMI_MODEL_THINKING_EFFORT=low` — but
the openai path expresses thinking as `reasoning_effort` (which models.json marks
`supportsReasoningEffort:false`) and injects no Qwen-native `enable_thinking`, so these
are likely no-ops for Qwen's native CoT, exactly as `MAX_THINKING_TOKENS` was for
claude. A clean green PASS is easiest to demonstrate against a **non-thinking model**
(e.g. Llama-3.3-70B, `EVAL_PROVIDER=llama`).

> `trace_stats.py` parses kimi's `agent.stream.jsonl` as if it were claude's
> stream-json (the schemas differ — kimi emits `{role,content,tool_calls}` objects),
> so its token/step counts read 0 for kimi until a `parse_kimi` branch is added. It's
> best-effort + per-round try/except, so this is non-fatal.

### claude-pro: subscription frontier model, off the gateway

`claude-pro` is Claude Code run against a **personal Anthropic subscription**
(OAuth to `api.anthropic.com`), not the gateway and not a self-hosted box.
It is the study's only frontier-model row (`claude-opus-4-8`) and exists to
anchor the open-weight results against a known-strong reference.

- **Auth is a token file, not a gateway key.** `agents/apptainer/claude-pro/.secrets/oauth_token`
  (from `claude setup-token`; path overridable via `OAUTH_TOKEN_FILE`). It plus all
  local scratch (`.setup-home/`, `.cache/`, `.work/`) are gitignored — never commit them.
- **Stage 0 differs** (see above): token-present + `api.anthropic.com`-reachable,
  not a gateway completion. `EVAL_MODELS` defaults to `claude-opus-4-8`.
- **Rate-limited by subscription session, not TPM.** Runs are **5 rounds/problem**
  (scored /5 and /20, not /10 and /40), one problem at a time with spacing between
  them — driven by `scripts/_orchestrate_claude_pro.sh` submitting
  `scripts/_run_one_claude_pro.slurm` per problem. A round that trips the session
  limit produces no `solution.json` (build-fail); those campaigns are re-run clean
  and excluded via `EXCLUDE_CAMPAIGNS` in `report_tables.py`.
- Phase 1 mirrors `claude/`; Phase 2 reuses `docker/swe-mini/measure.sh`.

### The solution.json contract

Every problem's `PROMPT.md` instructs the agent to write `solution.json` at the
problem root:

```json
{
  "build": ["make TOOLCHAIN=CLANG"],
  "run": ["./run-bench.sh -s 256 -v LolaBunnyGPU -c ./RabbitInput/Reference256.vol"],
  "variant_name": "LolaBunnyGPU"
}
```

- `build` / `run` — arrays of commands, replayed **in order** by the harness,
  relative to the problem root.
- `variant_name` — the identifier passed to `-v`.
- If the agent fails to produce a valid `solution.json`, the harness writes a
  FAIL result for that round.

### Evidence captured per round (inside `.eval/`)

- `facts.json` — raw receipts: agent, model, round, build/run commands, exit
  codes, timings. **No verdict, no derived metrics.**
- `build.log`, `run.log` — untouched command output.
- `gpu_info.txt` — `nvidia-smi -L` device identity.
- `gpu_usage.csv` — GPU utilization sampled at 250 ms during the run.
- `perf.log` — (perf-graded problems only) output of the harness bench:
  `BENCH mse=…` (agent-vs-cuBLAS correctness at the graded size) and
  `BENCH agent/cublas … ms_total=…` (cudaEvent timing).
- `kern_agent.csv` / `kern_cublas.csv` — (perf-graded problems only, when nsys is
  present) `nsys stats cuda_gpu_kern_sum` CSVs; the trusted timing source.
- `pass_result.json` — the verdict (overall `passed` + per-stage results),
  written **host-side** by `check_pass.py`.

Alongside `.eval/` in the round dir: `changed_diff.txt` (diff vs the pristine
problem), `solution.json`, the changed sources + any artifact, and the agent's
own trace — `agent.jsonl` (pi), `agent.traj.json` (swe-mini), or
`agent.stream.jsonl` (claude; Claude Code `--output-format stream-json`).

### Where output goes

```
traces/<problem>/<agent>/<model_tag>/<run_timestamp>/round-<N>/
results/runs.jsonl       one record per round
results/summary.json     pass@1 per (agent, model) + per-stage failure breakdown
```

The `<agent>` segment sits right after `<problem>` so `check_pass.py` (which
derives the problem from the first path segment) is unchanged, and pi/swe-mini
runs of the same model never collide. `<model_tag>` is the model id with `/`
replaced by `_`.

---

## Performance grading (tiled-gemm)

Most problems are graded on **correctness + GPU usage** only. `tiled-gemm` is an
optimization task, so it additionally grades **throughput** — and does so without
trusting any number the agent prints.

**The contract.** The agent is free to edit *any* file in the problem (split it,
rewrite the harness, change the algorithm) with exactly one fixed signature it
may not change: `void launchGemm(const float* dA, const float* dB, float* dC,
int N)` in `src/gemm.cuh`. That is the only thing the grader links against.

**How the number is produced (all harness-controlled, measure-time):**

1. `host.sh` binds `checks/tiled-gemm/bench.cu` → `/grade/bench.cu` read-only
   (only when that file exists — that's the gate that makes perf grading
   per-problem opt-in). It lives under `checks/`, so it never lands in the
   agent's workdir and the agent can't edit it.
2. `measure.sh`'s perf block compiles `/grade/bench.cu` together with the agent's
   kernel source(s) — `nvcc -O3 -arch=sm_86 -I src -lcublas`. The bench itself
   declares `extern void launchGemm(...)`, enforcing the interface.
3. The bench runs the agent kernel AND cuBLAS on one fixed, identical workload
   (`N=4096`, 50 iters, 5 warmup), brackets the timed loop with
   `cudaProfilerStart/Stop`, and prints a `BENCH mse=` (agent vs cuBLAS) plus
   cudaEvent `ms_total`. If `nsys` is present it profiles both modes
   (`--capture-range=cudaProfilerApi`) → `kern_agent.csv` / `kern_cublas.csv`.
4. `checks/tiled-gemm/check.py`'s `perf` stage grades on the **ratio**
   `cublas_ns / agent_ns` = agent throughput as a fraction of cuBLAS, preferring
   the nsys CSV kernel time and falling back to the cudaEvent `ms_total`. It
   first re-checks correctness (`BENCH mse` ≤ `BENCH_MSE_MAX`).

**Why a ratio.** It assumes *no* operation count — any `2·N³`-style constant
cancels — so the agent is free to find a cheaper algorithm, and the threshold is
GPU-independent (self-calibrates to whatever device runs it) instead of a brittle
per-GPU GFLOP/s number. Floor: `PERF_MIN_FRAC` (currently **0.15**, PROVISIONAL —
the naive starter measures ~12.6% of cuBLAS, so this just rejects the unoptimized
kernel; tune upward against real runs).

**nsys** is not in the CUDA base image; the agent Dockerfiles/defs add it from
NVIDIA's devtools apt repo as an isolated, non-fatal step. If it's missing the
check transparently falls back to cudaEvent timing (the ratio is still valid —
units cancel).

**This is tiled-gemm-only right now.** `stream-compact` and the RabbitCT variants
have no `bench.cu`, so their perf block is skipped and they stay correctness-only.
cuBLAS is GEMM-specific; another problem would need its own bench + reference.

### Smoke-testing the perf path — `scripts/smoke_perf.sh`

Runs Phase 2 (measure) only, with the pristine starter `src/gemm.cu` standing in
for an agent solution (no LLM). It checks an A40 is visible (aborts with an `srun`
hint otherwise) and reports whether nsys is in the image, copies `tiled-gemm` to a
tempdir, writes a `solution.json`, runs the apptainer measure container with
`measure.sh` + `bench.cu` bound, then prints `perf.log` / the kern CSVs and runs
`Check(...).evaluate()`. Use it after an image rebuild to confirm nsys is picked up
and the verdict is sourced from `nsys` rather than the cudaEvent fallback.

---

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `EVAL_AGENT` | `swe-mini` | Which agent harness (`swe-mini`, `claude`, `kimi`, or `pi`). pi is dropped from the current A40 run (no enforceable step cap). |
| `EVAL_MAX_TURNS` | `120` | claude/kimi only: the enforceable iteration cap (analogous to swe-mini's `step_limit`). claude → Claude Code `--max-turns`; kimi → config.toml `loop_control.max_steps_per_turn` (one turn in `-p` mode, so it bounds the whole run). |
| `EVAL_SKIP_PREFLIGHT` | `0` | Set `1` to skip the Stage-0 gateway reachability probe. |
| `EVAL_MODELS` | (all from `models.json`) | Comma-separated model IDs. Needs `jq` if unset. |
| `EVAL_ROUNDS` | `20` | Rounds per model. |
| `EVAL_PROBLEMS` | `RabbitCT-LolaBunny` | Comma-separated problem names. |
| `EVAL_SIZE` | `256` | Problem size (e.g. RabbitCT volume size). |
| `EVAL_VARIANT` | `LolaBunnyGPU` | Variant name for the solution. |
| `EVAL_SOLVE_PAR` | `4` | Max concurrent Phase-1 solve containers (pi only; shared GPU). `1` = serial. Phase 2 is always serial. |
| `MODELS_CONFIG` | `agents/apptainer/pi/models.json` | Gateway `baseUrl` + `apiKey` + model IDs. |
| `DATA_ROOT` | `<repo>/data` | Where bulk input data lives. |

---

## Pre-run smoke test (do this before every batch run)

Before committing to a long batch (especially after a reboot, a gateway
endpoint change, or a VPN switch), run these three quick checks in order. Each
costs at most one problem-round; they catch the failures that otherwise only
surface after the whole job list has burned GPU/gateway time.

**1. Endpoint reachable + tool-calling works.** The gateway IP/host and the
model id change depending on how the model is served (self-hosted sglang vs the
NHR endpoint), and a VPN can make a reachable host unreachable. Probe it from a
container so you test the *same* network path the agents use:

```bash
# Reachability + model list (substitute baseUrl/apiKey from models.json):
BASE=$(jq -r '.providers.local.baseUrl' agents/apptainer/pi/models.json)
KEY=$(jq -r '.providers.local.apiKey'  agents/apptainer/pi/models.json)
curl -s "$BASE/models" -H "Authorization: Bearer $KEY" | jq '.data[].id'

# Tool-calling actually parses (REQUIRED for pi — see Gotcha 4). Confirm
# choices[0].message.tool_calls is NON-empty (not leaked into content):
curl -s "$BASE/chat/completions" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' -d '{
    "model":"Qwen/Qwen3.6-35B-A3B-FP8",
    "messages":[{"role":"user","content":"run ls via the bash tool"}],
    "tools":[{"type":"function","function":{"name":"bash",
      "parameters":{"type":"object","properties":{"command":{"type":"string"}}}}}]
  }' | jq '.choices[0].message.tool_calls'
```

If `tool_calls` is empty but the call leaked into `content`, fix the sglang
`--tool-call-parser` (Gotcha 4) before running **pi**. `swe-mini` (litellm)
tolerates more, but a non-reachable endpoint fails both.

**2. swe-mini one-round smoke** — small problem, no bulk data, one round:

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- bash -lc 'cd /mnt/d/projects/nhr-agent-evaluation && \
  EVAL_AGENT=swe-mini EVAL_MODELS="Qwen/Qwen3.6-35B-A3B-FP8" \
  EVAL_ROUNDS=1 EVAL_PROBLEMS=tiled-gemm EVAL_SOLVE_PAR=1 bash scripts/run_eval.sh'
```

**3. pi one-round smoke** — same, swap the agent:

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- bash -lc 'cd /mnt/d/projects/nhr-agent-evaluation && \
  EVAL_AGENT=pi EVAL_MODELS="Qwen/Qwen3.6-35B-A3B-FP8" \
  EVAL_ROUNDS=1 EVAL_PROBLEMS=tiled-gemm EVAL_SOLVE_PAR=1 bash scripts/run_eval.sh'
```

`tiled-gemm` is the right smoke problem: CUDA but tiny, with **no bulk dataset**
(unlike RabbitCT's 4.5 GB input), so a round is fast. **Pass criterion:** the
run ends with a `[PASS]` line for the freshly-created trace
(`traces/tiled-gemm/<agent>/<model_tag>/<newest-timestamp>/round-1`). A
`[FAIL] (build)` with "No solution.json produced" means the agent never
executed a tool — re-check step 1 (endpoint/tool-calling), not the problem.

Pre-flight, also confirm Docker is up (Gotcha 2): from WSL, `docker info`
should succeed; if not, start Docker Desktop and wait for the daemon.

---

## Running / testing a single problem fast

To iterate quickly (one model, one round, one problem):

```bash
EVAL_MODELS=Qwen3.6-35B-A3B EVAL_ROUNDS=1 EVAL_PROBLEMS=tiled-gemm \
  bash scripts/run_eval.sh
```

### Re-judge existing traces — NO GPU, NO Docker

This is the big lever: after editing a threshold or a check, **re-run judgment
over existing evidence at zero GPU cost.** No agent re-run needed.

```bash
python3 checks/check_pass.py --traces traces                       # all problems
python3 checks/check_pass.py --traces traces --problems RabbitCT-LolaBunny
python3 checks/collect_results.py                                  # → results/
```

### Tuning / adding pass criteria

- **Tune a threshold:** edit the constants at the top of
  `checks/<P>/check.py` (e.g. `MSE_MAX`, `GPU_MIN_AVG`, `GPU_MIN_PEAK`), then
  re-run `check_pass.py`. No GPU.
- **GPU stage gates on PEAK only.** All problems now call the shared
  `gpu_stage(GPU_MIN_PEAK, GPU_MIN_AVG)`: it passes on `peak >= GPU_MIN_PEAK`
  and *reports* avg without gating on it. A correct GPU kernel with a long
  non-GPU tail (host I/O, setup) has low avg yet high peak and must still pass;
  peak is what rejects a CPU-only re-implementation. `GPU_MIN_AVG` is kept for
  telemetry/back-compat.
- **Every check ends with a non-gating `metric()` stage.** It calls
  `metric_stage(name, value, unit, higher_better)` to record the problem's
  rankable scalar (always `passed=True` — it records a number, it does not
  judge, so it can never flip a verdict). The payload is self-describing, so
  `collect_results.per_problem_rankings` / `report_tables.py` rank any problem
  without per-problem knowledge. tiled-gemm's `metric` and `perf` share one
  `_cublas_frac()` so they always report the same ratio.
- **Add a check:** create `checks/<P>/check.py` with a `Check(ProblemCheck)`
  subclass (model it on `checks/RabbitCT-LolaBunny/check.py`). Implement
  `stages()` returning ordered stage methods; mark `build`/`run` `fatal=True` so
  downstream stages are reported `skipped` (not `failed`) when a fatal stage
  breaks — this preserves *which stage broke*. End with a `metric()` stage.
  Use the inherited `facts()`, `log_value(pattern)`, `gpu_stats()`,
  `gpu_stage()`, and `metric_stage()` helpers — **never parse raw evidence
  files yourself.**

### Adding a new problem

1. `problems/<P>/` with a `PROMPT.md` (task + the `solution.json` contract) and
   build/run assets.
2. `checks/<P>/check.py` with the pass criteria (above).
3. Run with `EVAL_PROBLEMS="RabbitCT-LolaBunny,<P>"`.

### Adding a new agent

Add `agents/docker/<name>/` with a `Dockerfile`, a `host.sh` orchestrator, and
its in-container `solve.sh`/`measure.sh` pair — mirror pi or swe-mini, swapping
only the agent-invocation block in `solve.sh` (`measure.sh` is agent-agnostic
and can be copied verbatim). The container must write `.eval/facts.json` +
evidence + the agent trace. Run with `EVAL_AGENT=<name>`. Checks are reused unchanged.

### Adding a model

Either set `EVAL_MODELS` at runtime (comma-separated), or add it to
`agents/apptainer/pi/models.json` and let the runner auto-read it.

---

## What to never touch

- **`checks/` is the source of truth for verdicts and is agent-agnostic on
  purpose.** It lives outside the agent workspace so an agent can't game it.
  Don't make a check read the agent's self-reported result or parse raw files
  directly — go through the `eval_lib.py` accessors.
- **`facts.json` / the in-container scripts must not compute verdicts or
  metrics.** Side effects (build/run/sample) live in the container; *all*
  judgment is host-side. Keep that separation — it's what makes re-judging free.
- **Don't copy bulk data into the workdir.** `data/<P>/` is bind-mounted
  read-only; it's gitignored and ~4.5 GB for RabbitCT. The empty mountpoint left
  in the workdir is pruned by the trace rsync.
- **Don't delete `traces/`** casually — it's the expensive GPU output that
  re-judging depends on.
- **`agents/apptainer/` is legacy.** The active pipeline is Docker. (But note the
  `models.json` default path still points into `apptainer/pi/` — that one file
  is live.)
- **Keep `pi/host.sh` and `swe-mini/host.sh` in sync** on the shared logic
  (problem copy, data mount, chown-back, container-side rsync/diff trace
  extraction, check_pass). They differ only in agent-specific bits.

---

## Gotchas

**1. Run from WSL (Ubuntu-22.04), NOT PowerShell or Git-bash/msys.**
The pipeline is Linux shell + Docker. msys Git-bash mangles `/`-prefixed args
before they reach `docker` (e.g. `/bin/bash` → `/C:/Program Files/Git/usr/bin/bash`)
and can't bind-mount its temp paths through Docker Desktop — containers start but
the in-container scripts and mounts silently break. Real Linux paths
(`/mnt/d/...`, `/problem`, `/tmp`) only behave under WSL. PowerShell is only for
Windows-side things (starting Docker Desktop, editing settings).

**2. Docker Desktop must be running with WSL integration for Ubuntu-22.04.**
`AutoStart` is off, so after a reboot start it (PowerShell:
`Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"`) and wait
~5–30 s for the daemon. Inside WSL, `docker` is the Docker Desktop shim; if you
see "command could not be found in this WSL 2 distro", integration is off —
enable it in Docker Desktop settings (or `settings-store.json` →
`IntegratedWslDistros: ["Ubuntu-22.04"]`) and restart. `--gpus all` passthrough
works (verified with the local GTX 1050).

**3. The model gateway is external and its IP changes between sessions.**
`agents/apptainer/pi/models.json` holds the sglang `baseUrl` (e.g.
`http://10.0.89.x:30000/v1`) + `apiKey`. The IP is re-assigned every time the
endpoint is relaunched — check/update it before a run. Quick reachability test
from a container: `curl -s $baseUrl/models`.

**4. pi needs the sglang `--tool-call-parser` to MATCH the model's tool-call
format.** If it's wrong, the agent does one turn and stops with no
`solution.json` (→ `[FAIL] (build)`), because tool calls leak into `content` as
text and pi (which expects OpenAI structured `tool_calls`) executes nothing.
Qwen3.6-35B-A3B emits the Qwen3-Coder XML shape
`<tool_call><function=NAME><parameter=ARG>…</parameter></function></tool_call>`,
which needs `--tool-call-parser qwen3_coder` (NOT `qwen25`, which expects
Hermes-JSON). Verify before a full run by POSTing a `tools` payload to
`/v1/chat/completions` and confirming `choices[0].message.tool_calls` is
non-empty (not leaked into `content`).

**5. Trace extraction runs inside a container, not host-side, on purpose.**
The eval container runs as root and on WSL2 the host's view of the bind mount
lags the container's writes — a `stat` of `.eval/facts.json` can succeed while a
`readdir` of the workdir still omits that newest entry. `rsync`/`diff` discover
files via `readdir`, so host-side extraction intermittently produced
`.eval`-less, diff-only traces. So `host.sh` does the rsync/diff **in a
container** (whose view of the just-written workdir is coherent) and pipes the
result out as a tar; the host only *writes* `TRACE_DIR` from that byte stream.
The same container also chowns the workdir back so the EXIT-trap cleanup can
delete it. If you change this, preserve that invariant or you'll get flaky
missing-evidence failures.

**6. If you change a Dockerfile, rebuild with `--no-cache`.** The auto-build in
`host.sh` only triggers when the image tag is missing — it won't pick up
Dockerfile edits on an existing tag:

```bash
docker build --pull --no-cache -t agent-eval-pi:latest \
  -f agents/docker/pi/Dockerfile agents/docker/pi
# swap pi → swe-mini / agent-eval-swe-mini:latest for that image.
```

**7. `python3` is required on the host for the final checks.** If it's missing,
`host.sh` skips judgment with a warning and you must run `check_pass.py`
manually. `jq` is required only when `EVAL_MODELS` is unset.

**8. nsys lives in the agent image — rebuild to get it; absence is non-fatal.**
The perf grader prefers `nsys` kernel timing but falls back to cudaEvent
`ms_total` when nsys is absent (the ratio is still valid). An image built before
the nsys step was added will silently use the fallback. After (re)building, run
`scripts/smoke_perf.sh` and confirm the perf verdict says `(nsys)` not
`(cudaEvent)`. The nsys install is isolated + non-fatal on purpose so a devtools
-repo hiccup can't break the agent toolchain — but that also means a failed nsys
install is easy to miss; the smoke test is how you catch it.
