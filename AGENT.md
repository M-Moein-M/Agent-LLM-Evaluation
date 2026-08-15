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
  measure.sh                 In-container Phase 2, ONE canonical copy shared by
                             every agent and both runtimes (each host.sh binds it
                             as /measure.sh): replay solution.json's build/run →
                             .eval/ evidence + facts.json. NO judgment.
                             Records `node` in facts.json and .eval/host_info.txt
                             (lscpu/numactl) — the ranking metric is wall time on
                             a specific machine, so rounds are only comparable
                             within a node. GPU sampling is best-effort, so this
                             works unchanged on CPU-only test-cluster nodes.
                             Plus an OPTIONAL perf-grading block: if
                             /grade/bench.cu is bound (problem has one), compile
                             it against the agent kernel + cuBLAS and time both
                             (nsys, else cudaEvent) → .eval/perf.log + kern_*.csv.
  docker/                    CURRENT focus (easier GPU access).
    pi/                      pi agent (two-phase, parallel-capable):
      Dockerfile               CUDA base + Node + pi-coding-agent.
      host.sh                  Orchestrator: build image, build job list,
                               Phase 1 (solve, parallel) → Phase 2 (measure, serial),
                               extract traces, run host-side checks.
      solve.sh                 In-container Phase 1: run agent → source +
                               solution.json + agent.jsonl. NO measurement.
    swe-mini/                swe-mini agent (two-phase, mirrors pi):
      Dockerfile               CUDA base + Python + mini-swe-agent (litellm).
      host.sh                  Orchestrator (mirror of pi's).
      solve.sh                 In-container Phase 1: run mini agent → source +
                               solution.json + agent.traj.json. NO measurement.
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
                               Reuses agents/measure.sh for Phase 2.
      kimi/                  Kimi Code agent (apptainer-only — no docker twin yet):
        agent.def              CUDA devel base + Kimi Code native install
                               (KIMI_INSTALL_DIR=/usr/local → /usr/local/bin/kimi) + nsys.
        host.sh                Orchestrator (mirror of claude's apptainer host.sh).
        solve.sh               Phase 1: run Kimi Code headless against the NHR
                               gateway (OpenAI-compat — /v1 kept) → source +
                               solution.json + agent.stream.jsonl. Cap via config.toml
                               loop_control.max_steps_per_turn. Reuses
                               agents/measure.sh for Phase 2.
                             NOTE: models.json default still lives in apptainer/pi/.
      claude-pro/            Claude Code on a PERSONAL Anthropic subscription
                             (OAuth to api.anthropic.com — NOT the gateway,
                             never an open-weight model). The study's only
                             frontier-model row (claude-opus-5).
        agent.def              Same base as claude/, but auth is the subscription
                               token, not a gateway key.
        host.sh / solve.sh     Phase 1 mirrors claude/; Phase 2 reuses
                               agents/measure.sh. Scratch/secret dirs
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
  report_html.py             Same aggregation as report_tables.py, rendered as a
                             sortable, deterministic (no now()) standalone HTML
                             report + a hand-authored results/notes.yaml Notes
                             column. Pure stdlib + PyYAML.
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
                             still count on the A40. A campaign where EVERY round
                             failed before measure (no gpu_info.txt anywhere in it)
                             falls back to _gpu_by_problem: a device seen elsewhere
                             under the same problem, but ONLY if every sibling that
                             ever resolved one agrees — otherwise `unknown`. This is
                             what correctly leaves a genuinely CPU-only problem
                             (e.g. 2D-seq) unlabeled instead of the date-based
                             fallback mislabeling it A40.

data/<P>/                    Bulk per-problem input (GITIGNORED). Bind-mounted
                             read-only into the container, never copied per round.

traces/<problem>/<agent>/<model>/<timestamp>/round-<N>/   Output. See below.
results/                     Aggregated output: runs.jsonl + summary.json.
```

### Problems present

Graded (have a matching `checks/<P>/check.py`): **RabbitCT-LolaBunny** (CUDA GPU
backprojection, the default), **RabbitCT-LolaOMP** (OpenMP variant),
**tiled-gemm**, **stream-compact** (CUDA), **2D-seq** (CPU — 2D Poisson/SOR,
see below). Experimental / no checks yet: `MD-Bench`, `0`. The RabbitCT
variants share one ~4.5 GB `RabbitInput/` dataset.

`2D-seq` is the only **CPU-only** problem: no GPU, no dataset, no `--nv`
needed. It is a fixed-sweep-count SOR kernel graded on `solve()` wall time,
with a `DIGEST` line (printed by the frozen `src/verify.c`) gating correctness
at 1e-9 relative against `checks/2D-seq/reference.json`. Its `PROMPT.md` is
deliberately underspecified — it names no parallel programming model, no
hardware detail and no target machine — because it exists to test whether a
model can characterize an unfamiliar CPU on its own. Regenerate the reference
with `bash checks/2D-seq/make_reference.sh` after any `poisson.par` change.

Four files are frozen (`_FROZEN` in `checks/2D-seq/check.py`, enforced by the
`integrity` stage): `poisson.par`, `src/verify.c`, `src/verify.h` — and
`src/main.c`, which is frozen because it owns *both ends of the measurement*.
It brackets `solve()` with the timer and prints the `SOLVE_TIME` line the
metric stage ranks on. An editable driver lets a solution shrink the timed
region or hoist sweeps into setup and still emit a correct `DIGEST`: a better
score for no faster solver. Everything that legitimately needs changing
(`src/solver.c`, `src/allocate.c`, the build) is in other files.

The metric line is **tagged** (`SOLVE_TIME %.6fs`) rather than a bare
`" %.2fs"`. `eval_lib.log_value` takes the FIRST `re.search` match, so an
untagged number-plus-`s` pattern is hijacked by any timing the agent's own code
prints earlier in `run.log`. Microsecond resolution keeps a solver that gets
from ~100 s to well under a second still rankable.

### Running CPU problems on the NHR test cluster

CPU problems (e.g. 2D-seq) run on the test cluster instead of the A40 nodes.
A wave script targeting one of these differs from the A40 wave scripts only in
what the metric being wall time on one named machine forces:

- `--nodelist=cpu-node --exclusive`, no `--gres` — rounds are only comparable
  within a node, and nothing else may share it.
- No `--cpus-per-task`. On the A40 waves 16 CPUs capped compiles harmlessly;
  on a 288-CPU / 8-NUMA test-cluster node it would confine the agent to a
  16-CPU cgroup and hide the hardware it is supposed to discover.
- `EVAL_SOLVE_PAR=1` (see Phase 1/2 above).
- `EVAL_GPU=0`. Apptainer `--nv` is now conditional in every `host.sh`:
  `auto` (default) passes it only when `nvidia-smi` reports a GPU, `1`/`0`
  force it. On GPU nodes behavior is unchanged.

`measure.sh` records `node` in `facts.json` plus `.eval/host_info.txt`
(`lscpu`, `numactl -H`), so which machine produced a number is in the trace
rather than in whoever submitted the job. Note `PerfCounterAccess` is denied on
every test-cluster node probed so far — wall-clock only, no likwid/`perf stat`.

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
   defaults `EVAL_MODELS=claude-opus-5` for this agent (a real Anthropic id,
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

- **Image build.** `agents/apptainer/pi/agent.def` builds the local pi fork
  (adds `--step-limit`) FROM SOURCE inside the container at `apptainer build`
  time — `%post` runs `npm ci` + per-package `npm run build` + `npm pack` for
  all four `@earendil-works/pi-*` packages, then a single `npm install -g` of
  the resulting tarballs (must be one call so the cross-package deps resolve
  to each other locally, not the public registry). No `pi-dist/*.tgz` are
  committed anymore; the build needs outbound network access (npm registry +
  pi-ai's model-list fetch), so it must run from a node that has it.
- **Phase 1 — SOLVE (parallel, shared GPU).** Up to `EVAL_SOLVE_PAR` (default 4)
  `solve.sh` containers run at once. The agent's own compile/run iterations only
  need *correctness* feedback and mostly wait on the LLM gateway, so GPU
  contention is fine. Each writes source + `solution.json` + `agent.jsonl` to its
  workdir.
- **Phase 2 — MEASURE (serial, isolated).** `agents/measure.sh` replays each
  solution's build/run with 250 ms GPU sampling, producing the graded evidence.
  Strictly serial so each measured run owns the GPU — this is what timing/GPU
  thresholds are judged on. Set `EVAL_SOLVE_PAR=1` for fully-serial behavior.
- **CPU problems need `EVAL_SOLVE_PAR=1`, not just a serial Phase 2.** The
  *scored* number is safe either way, since Phase 2 is always serial. But in
  Phase 1 the agents time their own experiments to decide what to optimize, and
  N concurrent agents on one CPU node make every one of those timings garbage —
  which corrupts the agents' reasoning, i.e. the thing under study. On GPU
  problems the agents lean on correctness feedback instead, so overlap is fine.

### swe-mini: same two-phase split

swe-mini now mirrors pi: `solve.sh` (Phase 1, runs the mini agent → source +
solution.json + agent.traj.json, parallelizable up to `EVAL_SOLVE_PAR`) and
`agents/measure.sh` (Phase 2, replay build/run with GPU sampling, strictly
serial). Phase 2 is where agents stop differing, so there is exactly one
`measure.sh` for all of them — the previous per-agent copies were identical
apart from a default env var and a comment, and had to be fixed twice.

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

### `-think` model ids vs the bare id (self-hosted DeepSeek-V4-Flash-0731)

`deepseek-ai/DeepSeek-V4-Flash-0731-think` and `deepseek-ai/DeepSeek-V4-Flash-0731`
are **the same weights on the same sglang backend**. They differ only in the
LiteLLM deployment in front of it:

| | bare id | `-think` id |
|---|---|---|
| server-side reasoning | off (LiteLLM default) | on, via `extra_body.chat_template_kwargs.thinking: true` |
| where reasoning arrives | separate `reasoning_content` field | folded into `content` as `<think>…</think>` (`merge_reasoning_content_in_choices: true`) |
| `model_info.supports_reasoning` | — | `false`, to keep LiteLLM off the `/v1/responses` bridge |
| used by | kimi, swe-mini, pi | claude only |

**Why the split.** The OpenAI-compat harnesses (kimi, swe-mini, pi) want the clean
`reasoning_content` split and read it fine, so they take the bare id. Claude Code
speaks Anthropic `/v1/messages`, and LiteLLM's Anthropic SSE adapter emits
malformed thinking blocks for a reasoning model — Claude Code dies on turn 1 with
*"Content block is not a thinking block"*. Merging the reasoning into `content`
means no thinking block is ever emitted, so the alias is a **transport
workaround**, not a different model or a different amount of reasoning. Full
LiteLLM config details in the `-think` deployment: see the three keys documented
in `models.yaml` (the third, `supports_reasoning: false`, is the non-obvious one —
without it the request gets re-routed to `/v1/responses`, which sglang does not
serve).

**Consequences for results.** Because the checkpoint is identical, the runner sets
`EVAL_MODEL_TAG=deepseek-ai_DeepSeek-V4-Flash-0731` so all four agents' traces
land under one folder. `facts.json` still records the **wire** id, so the
`-think` runs would otherwise appear as a separate model row; `slides/make_figures.py`
canonicalizes `DeepSeek-V4-Flash-0731-think` → `DeepSeek-V4-Flash-0731` in the
`ROWS_LATEST` builder for exactly this reason. Per-model aggregation must never be
split by the alias.

**On the hub there is no `-think` alias** (the hub serves the 0731 weights
under the bare tag `deepseek-ai/DeepSeek-V4-Flash`; the `-0731` id itself 403s).
Claude against the hub therefore needs `EVAL_MAX_THINKING_TOKENS=0` to survive
turn 1 — see `scripts/_run_agent_dsflash0731.slurm`, mode `hub`.

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
- **Compaction is disabled for fairness** (since 2026-08-10). Kimi's
  `micro_compaction` is ON by default, which silently summarizes older turns as
  history nears the context limit and lets a run continue where pi/swe-mini —
  which have no compaction — simply die. Job 3973505 made the asymmetry concrete:
  swe-mini hit `Input length (216093 tokens) exceeds the maximum allowed length`
  **11 times** (one propagating as an unhandled `OpenAIError` that killed the run)
  against **zero** for kimi on the same model and problems. `solve.sh` now exports
  `KIMI_CODE_EXPERIMENTAL_MICRO_COMPACTION=false`, matching claude's
  `DISABLE_AUTO_COMPACT=1` policy, so context exhaustion is a real failure — the
  analog of running out of steps — for every agent. `EVAL_KIMI_COMPACTION=1`
  restores the CLI default.
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
It is the study's only frontier-model row (`claude-opus-5`) and exists to
anchor the open-weight results against a known-strong reference.

- **Auth is a token file, not a gateway key.** `agents/apptainer/claude-pro/.secrets/oauth_token`
  (from `claude setup-token`; path overridable via `OAUTH_TOKEN_FILE`). It plus all
  local scratch (`.setup-home/`, `.cache/`, `.work/`) are gitignored — never commit them.
- **Stage 0 differs** (see above): token-present + `api.anthropic.com`-reachable,
  not a gateway completion. `EVAL_MODELS` defaults to `claude-opus-5`.
- **Rate-limited by subscription session, not TPM.** Runs are **5 rounds/problem**
  (scored /5 and /20, not /10 and /40), one problem at a time with spacing between
  them — driven by `scripts/_orchestrate_claude_pro.sh` submitting
  `scripts/_run_one_claude_pro.slurm` per problem. A round that trips the session
  limit produces no `solution.json` (build-fail); those campaigns are re-run clean
  and excluded via `EXCLUDE_CAMPAIGNS` in `report_tables.py`.
- **No compaction, and no window override — a DELIBERATE asymmetry.**
  `DISABLE_AUTO_COMPACT=1` keeps compaction off, and the window is left at the
  model's native size (1M on a current Opus subscription), so total tokens per round
  run as high as the model allows. The gateway agents are hard-capped at the served
  model's `max_model_len` of 262144, so claude-pro has ~4x their context. This is
  intentional: the open-weight rows have a real server-side limit, and the study
  reports that limit rather than handicapping the frontier row to match it.
  **State the asymmetry wherever these rows appear side by side.**

  **`CLAUDE_CODE_AUTO_COMPACT_WINDOW` must not be set anywhere in this repo, at any
  value.** It is a token count, not a compaction toggle (`DISABLE_AUTO_COMPACT` is
  the toggle), so any value clamps the window — including `0`, which is not
  "unlimited". There is deliberately no `EVAL_*` knob for it; nothing to
  misconfigure and nothing to forward.

  **Why the explicit set instead of trusting `--cleanenv`:** the 20260728 Opus-5
  wave lost **all 5 rounds** to `"Prompt is too long"` at peak contexts of
  168857–175902 — a 200k wall. Cause was our own environment: `~/.bashrc` on the
  submitting host exported `CLAUDE_CODE_AUTO_COMPACT_WINDOW=200000`, Slurm's
  `#!/bin/bash -l` sourced it, and `apptainer exec` (which inherits host env by
  default) carried it into the container. With `DISABLE_AUTO_COMPACT=1` that clamp
  is a hard stop rather than a compaction. **That wave measured a config leak, not
  the model — do not cite it as a capability result.**

  **How to tell a client clamp from a server refusal** (this cost a full wave and
  ~$25 to work out): in `agent.stream.jsonl`, a client-side refusal ends with an
  assistant message whose `"model"` is `"<synthetic>"` and a `result` event with
  `"api_error_status": null` — the request was never sent, so there is no HTTP 400
  and the `request_id`s present belong to earlier successful calls. A genuine
  entitlement refusal instead returns `400 invalid_request_error` with its own
  `request_id` and a message naming the plan. Check those two fields **first**;
  probing env vars is the slow way round and is easy to get wrong (a probe run
  through a login shell with `HOME` preserved re-sources `~/.bashrc` and silently
  restores the very variable you unset).

  **Fix landed:** every apptainer `host.sh` (`pi`, `swe-mini`, `claude`,
  `claude-pro`) now passes `--cleanenv` to `apptainer exec`, which blocks host-env
  inheritance outright instead of relying on nobody's `~/.bashrc` exporting the
  wrong thing. The tradeoff: nothing implicit reaches the container anymore, so
  every var a script needs (`EVAL_*`, `EVAL_EFFORT`, etc.) must be forwarded
  explicitly via `--env` — each `host.sh` does this already; if you add a new
  `EVAL_*` knob, forward it there too or it silently reads as unset inside.
- Phase 1 mirrors `claude/`; Phase 2 reuses `agents/measure.sh`.

### Packing multiple campaigns onto one GPU

A wave script can run 2-3 whole agent campaigns (different agents/models/providers)
as background jobs inside ONE SLURM allocation instead of one GPU reservation
per campaign — Phase 1 (solve) is already GPU-contention-tolerant (see the pi
rationale above; every agent shares it), so the only thing packing threatens is
Phase 2 (measure), where wall-clock IS the score. Every apptainer `host.sh`
(`claude`, `pi`, `swe-mini`) guards this with a `flock` (`MEASURE_LOCK`, keyed
off `$TMPDIR`) around both the measure step and the trailing
`check_pass.py`/`trace_stats.py` regen, so concurrent `host.sh` instances —
even different agents — packed into the same job still serialize exactly
where isolation matters. Set `MEASURE_LOCK` to the same path across a mixed
wave (e.g. a merged kimi+pi run) to share one lock; leave it at the per-job
`$TMPDIR` default when campaigns don't need to cross-serialize. Set env vars
with plain `export` inside a packing script, not `sbatch --export=...` —
`--export` splits on every comma, which silently truncates a comma-separated
`EVAL_PROBLEMS` to its first entry.

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
| `EVAL_SOLVE_PAR` | `4` | Max concurrent Phase-1 solve containers (pi only; shared GPU). `1` = serial. Phase 2 is always serial. Use `1` for CPU problems — see "How a run works end-to-end". |
| `EVAL_GPU` | `auto` | Apptainer GPU passthrough. `auto` = pass `--nv` only if `nvidia-smi` reports a GPU; `1` force on, `0` force off (CPU-only test-cluster nodes). |
| `EVAL_PROVIDER` | `local` | Selects `providers.<name>` in `MODELS_CONFIG` (e.g. `local` = hub, `llama`/`gemmasglang`/`osslocal`/`kimi3` = other self-hosted or separately-keyed endpoints). |
| `EVAL_EFFORT` | `high` | claude/claude-pro only: `claude --effort <level>`. Real lever for claude-pro (Anthropic model); kept identical for claude against the gateway for a clean comparison, but a no-op there — the gateway models declare `supportsReasoningEffort:false`. |
| `MODELS_CONFIG` | `agents/apptainer/pi/models.json` | Gateway `baseUrl` + `apiKey` + model IDs. The tracked file carries the real `baseUrl` but a placeholder `apiKey` (`PASTE_KEY_IN_models.local.json`) — real keys go in the gitignored sibling `agents/apptainer/pi/models.local.json`, which every wave/smoke script points `MODELS_CONFIG` at explicitly. A forgotten key then fails fast (401) instead of a real one landing in git. |
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
# Reachability + model list (substitute baseUrl/apiKey from models.json).
# NOTE: jq is NOT installed on the test-cluster LOGIN node (login-node), so run
# this probe there with python3 instead. It IS present on the cpu-node compute node
# (confirmed 2026-07-27: run_eval.sh's Stage-0 preflight took its jq branch and
# printed "[ok] ... (HTTP 200)" inside a batch job). Do not assume either way for
# a node you have not checked — where jq is missing, Stage 0 WARN-skips silently
# and this manual probe becomes the only gate a wave gets:
#   eval "$(python3 -c 'import json;c=json.load(open("agents/apptainer/pi/models.json"))["providers"]["local"];print(f"BASE={c[\"baseUrl\"]}\nKEY={c[\"apiKey\"]}")')"
# If the port refuses the connection while `ip route get <ip>` shows a route,
# the gateway has moved or is down — a self-hosted sglang IP is re-assigned per
# session. The stable endpoint is the hub: https://gateway.example.org/v1
# (LiteLLM proxy over vLLM). Verified 2026-07-27 from login-node, with the key in
# the untracked agents/apptainer/pi/models.local.json (MODELS_CONFIG points at it):
#   * /v1/models -> 200, 13 models served CONCURRENTLY (unlike sglang's one-at-a-
#     time; pinning every claude tier slot to EVAL_MODEL is still what you want,
#     it just stops being mandatory).
#   * /v1/chat/completions -> 200, and tool_calls come back STRUCTURED, not
#     leaked into content (Gotcha 4 satisfied for deepseek-ai/DeepSeek-V4-Flash).
#   * /v1/messages -> 200 including stream:true with tool_use +
#     input_json_delta, so the Anthropic-protocol claude agent works against it.
#   * max_tokens 65536 accepted (1048576 is not a real cap — it exceeded the
#     model's own contextWindow).
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
its in-container `solve.sh` — mirror pi or swe-mini, swapping only the
agent-invocation block. Do NOT add a `measure.sh`: bind the shared
`agents/measure.sh`, as every existing host.sh does. The container must write
`.eval/facts.json` +
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

**9. Stage a job workdir with `rsync -a`, never `cp -a`.** All four apptainer
`host.sh` scripts copy the problem into a per-job workdir this way. `cp -a`
tries to replicate ACLs, and the test-cluster filesystem rejects that —
"preserving permissions ... Operation not supported" on every single file,
~25 lines of noise per round. `rsync -a` preserves mode/mtime/symlinks without
touching ACLs. mtime matters beyond the noise: `measure_job`'s
`rsync --compare-dest` decides "did the agent modify this file?" from
size+mtime, so a copy that reset mtimes would flag every untouched file as
agent-modified.

**10. kimi-code crash-loops on Kimi-K2.x's own `</think>`-omission bug; claude
and swe-mini don't, for unrelated client-side reasons — don't mistake either
outcome for a harness bug.** Root cause is **model-level**, confirmed across
three independent serving stacks (sglang, vLLM, Azure AI Foundry — including a
report on kimi-code's own tracker, `MoonshotAI/kimi-code#520`): the Kimi K2.x
checkpoint intermittently fails to emit its closing `</think>` token, worst on
the turn right after a tool call. Nothing server-side is broken; the model
occasionally just doesn't close its own thinking block.

What that does to each harness depends entirely on how sglang's
`--reasoning-parser` interacts with the missing tag, and how tolerant each
client is of the result:

- **sglang's reasoning-parser has no fallback for a missing `</think>`.** It
  dumps the *entire* raw generation into `reasoning_content` and hands the
  tool-call-parser an empty `text`. The tool-call-parser then finds nothing,
  so the final response is `content=""` + `tool_calls=[]` — thinking is
  present, actionable output is not (`adapter.py`'s reasoning-parser runs
  *before* the tool-call-parser and unconditionally overwrites `text`).
- **kimi-code's client explicitly checks for "text or tool_calls, else fail."**
  Hitting the empty case above raises a hard `provider.api_error`
  (`APIEmptyResponseError`) after retries — one bad turn kills the whole run.
  Confirmed via a real (non-simulated) smoke run against
  `10.0.0.1:30000` with `--reasoning-parser` dropped: crash disappears
  (raw reasoning merges verbatim into `content`, literal `</think>` shows up
  as plain text), but the model can still return a near-empty turn
  (`content: "</think>"`) — a milder, unfixable-by-config residual.
- **claude (Anthropic-protocol, via the litellm-based gateway) doesn't
  crash on this**, not because it avoids the sglang bug (it doesn't — same
  bug, same backend), but because **no fail-fast check exists anywhere in the
  litellm→claude-code path.** litellm passes the (already-empty) OpenAI
  response through as a thin/near-empty Anthropic turn; claude-code's own
  turn loop doesn't treat that as fatal either. Confirmed on a real 40-job
  production run (`claude` + `moonshotai/Kimi-K2.7-Code`, 20/40 pass): zero
  matches for `APIEmptyResponseError`/`thinking content`/`retry` in the full
  console log, `agent.stderr.log` empty for a sampled FAIL round, and that
  round ran its full tool-call sequence before ending on `error_max_turns`
  (turn-budget exhaustion), not an API error. All 20 FAILs are genuine
  build/mse/run failures, not this bug.
- **swe-mini (mini-swe-agent + litellm, direct OpenAI-compat) also doesn't
  crash, via yet another path — and it's the only harness of the four that
  recovers instead of merely not-crashing.** A turn with no tool calls is a
  *recoverable format error*, never a completion signal. See Gotcha 11 for the
  full mechanism and source citations; this is the single biggest design
  difference between swe-mini and kimi-code. (Earlier revisions of this note
  said mini parses actions out of response *text* (a bash code block). That is
  the `mini_textbased.yaml` / `litellm_textbased_model` path — **not** what our
  runs use. `host.sh` binds `mini.yaml`, which is the structured-`tool_calls`
  path via `litellm_model`. The tolerance conclusion holds either way, but for
  a different reason than originally written.)
- **pi doesn't crash either — but do NOT read that as "pi recovers."**
  `agents/apptainer/pi/pi-edited/packages/ai/src/api/openai-completions.ts:1006-1016`
  explicitly skips assistant turns with no `content` and no `tool_calls` when
  building the next request — comment: *"This handles aborted assistant
  responses that got no content."* That prevents an API-request *serialization*
  crash; it does not keep the agent working. pi's loop
  (`packages/agent/src/agent-loop.ts:172`) is
  `while (hasMoreToolCalls || pendingMessages.length > 0)`, and a turn with zero
  tool calls sets `hasMoreToolCalls = false` — so pi exits the loop and ends the
  session on a degenerate turn, structurally the same as kimi-code. Confirmed by
  reading the vendored source, not inferred.

**Net: this is a two-layer problem, and only one layer has a server-side
fix.** Dropping `--reasoning-parser kimi_k2` fixes layer 1 (sglang no longer
force-empties `content` when `</think>` is missing) — necessary for everyone.
Layer 2 is client-side tolerance of a still-possible degenerate turn (the
model can independently emit near-empty output regardless of the parser
config): pi and swe-mini already tolerate it by design, claude/litellm never
validates for it, but **kimi-code has a hard "text or tool_calls, else fail"
check baked into its own client with no server-side override.** So: pi runs
clean with the parser dropped; kimi-code does not — there is no serving-stack
setting that patches a client-side design decision. Fixing kimi-code would
require a change in kimi-code itself (or a proxy that filters/retries
degenerate responses before kimi-code's own SDK sees them), not a sglang flag.

**Layer 3 exists too, and patching layer 2 exposes it — see Gotcha 11.** Making
kimi-code stop *crashing* on a degenerate turn does not make it *finish the
task*: it then accepts the degenerate turn as a finished answer and exits 0 with
no work done. Only swe-mini survives layer 3.

**Checked: kimi-code 0.34.0 does NOT fix this (0.31.0 is what `agents/apptainer/kimi/`
builds; 0.34.0 built separately at `agents/apptainer/kimi-034/agent.sif`, confirmed
via `kimi --version`).** Cloned `MoonshotAI/kimi-code` to a scratch dir and diffed
`@moonshot-ai/kimi-code@0.31.0` against `@moonshot-ai/kimi-code@0.34.0` (92 commits,
2357 files changed repo-wide, so the diff tooling is confirmed working):
- `packages/kosong/src/generate.ts` and `errors.ts` — the actual throw-site for
  `APIEmptyResponseError` (both the fully-empty and thinking-only checks) — are
  **byte-for-byte unchanged**. Zero diff.
- `packages/agent-core-v2/src/kosong/{contract,protocol}/errors.ts` did change, but
  it's a pure refactor (error classes made "born-coded" — each computes its own wire
  code at construction instead of a central switch in `translateProviderError`).
  Classification is identical before/after: `APIEmptyResponseError` still maps to
  `provider.api_error` (or `provider.filtered` iff `finishReason === 'filtered'`),
  and that code is still absent from the protocol-level `retryable` list in both
  versions. The `isRetryableGenerateError(APIEmptyResponseError) === true` test
  assertion (agent-core-v2 llmRequester layer) is also identical in both versions —
  so kimi-code already retries this error internally today, and 0.34.0 changes
  nothing about the retry count, backoff, or classification.
- `CHANGELOG.md` across `kimi-code`/`kosong`/`agent-core`/`agent-core-v2` for the
  92-commit range: no entry mentions empty-response, thinking-only, or
  reasoning-parsing fixes. Only tangential retry-adjacent entries are a kimi-web UI
  cosmetic change (shows "attempt N of M" during automatic retries) and an unrelated
  compaction token-counting fix.

Conclusion: no version bump alone fixes this — the hard "text or tool_calls, else
fail" check is unchanged in the CLI's newest release, so the only mitigations remain
the ones above (drop `--reasoning-parser`, and/or don't run kimi-code against
Kimi-K2.x until Moonshot patches the client itself).

**11. How a harness decides "the agent is done" is a real capability
difference, and swe-mini is the only one of the four that gets it right.**
This is the deepest of the harness differences and it is *independent of the
Kimi bug* — Gotcha 10 is just the cleanest way to observe it. Two designs:

- **Completion by absence (kimi-code, pi, claude).** A turn with no tool calls
  IS the completion signal. The loop cannot distinguish "the model is genuinely
  finished" from "the model emitted garbage" — both look identical: text
  present, `tool_calls` empty. kimi-code:
  `packages/agent-core-v2/src/agent/loop/loopService.ts`. pi:
  `packages/agent/src/agent-loop.ts:172` (`while (hasMoreToolCalls || ...)`).
- **Completion by explicit positive signal (swe-mini).** Nothing implicit ever
  ends a run. Verified in the installed source inside
  `agents/apptainer/swe-mini/agent.sif` (mini-swe-agent 2.4.4):
  - No tool calls → `FormatError`, not completion:
    `minisweagent/models/utils/actions_toolcall.py:40` — *"No tool calls found
    in the response. Every response MUST include at least one tool call."*
    `DefaultAgent.run` (`agents/default.py`) catches it, appends the corrective
    message via `format_error_template`, and keeps looping.
  - The ONLY way to finish is the sentinel, and it is checked **in the
    environment, on the executed command's stdout** — not in the model's prose:
    `minisweagent/environments/local.py:_check_finished` raises `Submitted` iff
    the first line of output is `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` AND
    `returncode == 0`. The model must actually *run* the echo; writing the
    string in its reply does nothing. (Same check in `docker.py`,
    `singularity.py`, `bubblewrap.py`, etc.)
  - It does not spin forever: `max_consecutive_format_errors: 3` (AgentConfig
    default; our `mini.yaml` does not override) exits with a distinguishable
    `RepeatedFormatError` status, and the counter resets after any clean step
    (`self.n_consecutive_format_errors = 0`). Plus `step_limit: 120`,
    `cost_limit`, `wall_time_limit_seconds`.

**Why it matters — measured, not theoretical.** Against a self-hosted
`moonshotai/Kimi-K2.7-Code` (cluster multi-node sglang, `10.0.0.1:30000`), the
patched `kimi-edited` image on `problems/2D-seq`, 4 parallel rounds: **4/4
identical silent no-ops.** Each round did one big parallel-`Read` batch, then
returned a turn whose entire `content` was the literal leaked string
`"</think>"` with no tool calls. kimi-code took that as the final answer and
ended the session — `exit=0`, `agent.stderr.log` **0 bytes**, no
`solution.json`, no edit, no build. A grader keying on exit status or stderr
sees a clean run. (Note this turn has `hasText=true`, so it bypasses the
Gotcha-10 `hasThink && !hasText` guard entirely — the layer-2 patch is inert
here. This is layer 3.)

Under swe-mini's design the same model output would have produced a corrective
message and three more attempts, then exited `RepeatedFormatError` — a visible,
attributable failure instead of a fake pass.

**Consequences for this eval:**
- **Do not compare pass rates across harnesses without checking exit reasons.**
  Completion-by-absence harnesses can convert a model failure into a silent
  `exit=0` no-op. Cross-check that `solution.json` exists and the workdir was
  actually modified; `exit=0` + empty stderr is NOT evidence of a real attempt.
- If kimi-code is to be fixed properly, the port is swe-mini's contract, not a
  Kimi-specific text filter: *absence of a tool call is a recoverable format
  error; completion requires an explicit positive signal; cap consecutive
  failures and exit with a distinct status.* That is a change to shared loop
  semantics in `loopService.ts` affecting every model — scope it deliberately.
