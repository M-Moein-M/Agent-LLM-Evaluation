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
    swe-mini/                swe-mini agent (two-phase, mirrors pi):
      Dockerfile               CUDA base + Python + mini-swe-agent (litellm).
      host.sh                  Orchestrator (mirror of pi's).
      solve.sh                 In-container Phase 1: run mini agent → source +
                               solution.json + agent.traj.json. NO measurement.
      measure.sh               In-container Phase 2: replay build/run with GPU
                               sampling → .eval/ evidence + facts.json. NO judgment.
      mini.yaml                Prompt templates + finite step_limit.
  apptainer/                 LEGACY (pi, claude). Not part of the Docker pipeline.
                             NOTE: models.json default still lives here (see below).

problems/<P>/                One directory per problem.
  PROMPT.md                  Task description + the solution.json contract the
                             agent must produce. Fed to the agent in solve.sh.
  (sources, Makefile, mk/, run-bench.sh, ...)   per-problem build/run assets.

checks/                      Agent-AGNOSTIC judging subsystem (outside agent workspace
                             so it can't be gamed). Host-side, python3 only.
  eval_lib.py                ProblemCheck base + evidence accessors:
                             facts(), log_value(pattern), gpu_stats().
  <P>/check.py               Per-problem pass criteria: ordered stages + threshold
                             constants at top of file.
  check_pass.py              Driver: runs each problem's check over a trace →
                             .eval/pass_result.json.
  collect_results.py         Aggregates verdicts → pass@1 per (agent, model).

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

`scripts/run_eval.sh` is two-stage and selects the agent via `EVAL_AGENT`
(default `pi`):

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
- `pass_result.json` — the verdict (overall `passed` + per-stage results),
  written **host-side** by `check_pass.py`.

Alongside `.eval/` in the round dir: `changed_diff.txt` (diff vs the pristine
problem), `solution.json`, the changed sources + any artifact, and the agent's
own trace — `agent.jsonl` (pi) or `agent.traj.json` (swe-mini).

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

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `EVAL_AGENT` | `pi` | Which agent harness (`pi` or `swe-mini`). |
| `EVAL_MODELS` | (all from `models.json`) | Comma-separated model IDs. Needs `jq` if unset. |
| `EVAL_ROUNDS` | `20` | Rounds per model. |
| `EVAL_PROBLEMS` | `RabbitCT-LolaBunny` | Comma-separated problem names. |
| `EVAL_SIZE` | `256` | Problem size (e.g. RabbitCT volume size). |
| `EVAL_VARIANT` | `LolaBunnyGPU` | Variant name for the solution. |
| `EVAL_SOLVE_PAR` | `4` | Max concurrent Phase-1 solve containers (pi only; shared GPU). `1` = serial. Phase 2 is always serial. |
| `MODELS_CONFIG` | `agents/apptainer/pi/models.json` | Gateway `baseUrl` + `apiKey` + model IDs. |
| `DATA_ROOT` | `<repo>/data` | Where bulk input data lives. |

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
- **Add a check:** create `checks/<P>/check.py` with a `Check(ProblemCheck)`
  subclass (model it on `checks/RabbitCT-LolaBunny/check.py`). Implement
  `stages()` returning ordered stage methods; mark `build`/`run` `fatal=True` so
  downstream stages are reported `skipped` (not `failed`) when a fatal stage
  breaks — this preserves *which stage broke*. Use the inherited `facts()`,
  `log_value(pattern)`, and `gpu_stats()` helpers — **never parse raw evidence
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
