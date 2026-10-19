# AGENT.md

Guide for AI coding agents working in this repo. Read it first, then change what the task needs.
For a human overview see README.md.

**Maintenance contract.** Any change to a directory, script, environment variable, output file or
schema must update this file (and `docs/results-schema.md` for outputs) in the same change. Do not
leave it stale.

## What this repo is

A harness that gives coding agents (claude, kimi, pi, swe-mini) the same programming problems, runs
them against any OpenAI-compatible endpoint, replays and grades each solution on the host, and
aggregates pass rates, token use and failure causes. It also holds the serving stack used to host
open-weight models (`serving/`) and the original author's results (`reference-results/`).

## Map

| Path | What it is | Needed to run evaluations |
|---|---|---|
| `endpoints.yaml`, `.env` | Your endpoints, models and API keys. Created from `endpoints.yaml.example` and `.env.example`. Gitignored. | yes |
| `scripts/run_eval.sh` | Single entry point. Resolves the endpoint, probes it, prepares data, runs `agents/apptainer/$EVAL_AGENT/host.sh`. | yes |
| `scripts/run_eval.slurm` | SLURM wrapper for one agent on one GPU. | optional |
| `scripts/lib/resolve_endpoint.py`, `endpoint.sh` | Turns `endpoints.yaml` + `.env` into `.run/models.json` and exports. Also the protocol-aware preflight. | yes |
| `scripts/remeasure.sh`, `smoke_perf.sh`, `check_starter.sh` | Replay stored solutions on a quiet GPU. Smoke-test the perf grader. Run a problem's starter code through its checker. | optional |
| `agents/apptainer/<agent>/` | `agent.def` (image recipe), `host.sh` (orchestrator), `solve.sh` (runs inside the container). Agents: `claude`, `kimi`, `pi`, `swe-mini`, `claude-pro`. | yes |
| `agents/measure.sh` | Phase 2, shared by all agents: replays `solution.json` and records evidence in `.eval/`. No judgment. | yes |
| `problems/<P>/` | One directory per problem: `PROMPT.md`, starter code, `Makefile`. Copied into the agent workspace. | yes |
| `problems/_template/` | Copy-and-edit starter for a new problem (CPU toy, builds with gcc). | for extending |
| `checks/` | Host-side grading, outside the agent workspace so it cannot be read or gamed. `eval_lib.py`, `check_pass.py`, `trace_stats.py`, `failure_class.py`, `<P>/check.py`, `_template/`. Flat import root: scripts do `import failure_class`. | yes |
| `tests/endpoint/` | Endpoint test suite: protocol probes plus one real round per agent. | recommended |
| `tools/summarize.py` | pass@1 and median tokens per (problem, agent, model) from `traces/`. | optional |
| `tools/collect_results.py` | Aggregates `traces/` into `results/runs.jsonl` and `summary.json`. | optional |
| `tools/trace-vis/` | Local web UI for reading trajectories (Apptainer image). | optional |
| `serving/` | SGLang + LiteLLM + Prometheus/Grafana stack for self-hosting models on SLURM. Independent of the harness. | no |
| `docs/` | `EXTEND_PROMPT.md` (prompt for your agent), `gotchas.md`, `results-schema.md`. | no |
| `reference-results/` | The original author's results: figures, aggregates, analysis scripts, trace manifest. **Safe to delete.** Nothing outside it imports from it. | no |
| `traces/`, `results/`, `eval-logs/`, `data/`, `.run/` | Generated. Gitignored. `traces/` is the expensive output, do not delete it casually. | generated |

## How a run works

```
scripts/run_eval.sh
  resolve endpoints.yaml -> .run/models.json, EVAL_PROVIDER, EVAL_MODELS
  Stage 0: 1-token probe per model (skip: EVAL_SKIP_PREFLIGHT=1)
  Stage 1: bulk input data for RabbitCT problems (data/RabbitCT)
  exec agents/apptainer/$EVAL_AGENT/host.sh
      Phase 1 (parallel, EVAL_SOLVE_PAR): per (model, problem, round) an Apptainer container runs solve.sh.
          The agent works in a copy of problems/<P>, writes solution.json, its trace goes to agent.*.
          A watchdog kills a solve whose trace file stops changing (EVAL_SOLVE_STALL_TIMEOUT).
      Phase 2 (serial, flock): measure.sh replays solution.json (build, run, GPU sampling, perf bench)
          and writes .eval/facts.json + logs. Serial because timing is graded.
      Host: checks/check_pass.py -> .eval/pass_result.json, checks/trace_stats.py -> .eval/stats.json
traces/<problem>/<agent>/<model-tag>/<campaign>/round-<N>/
```

Why the split: the agent's own compile/run loop only needs correctness feedback, so solves share a
GPU. The graded measurement needs an isolated GPU, so it runs alone afterwards.
Why checks are on the host: `facts.json` records what happened, never a verdict. Verdicts are
recomputed by `python3 checks/check_pass.py --traces traces` at no GPU cost after editing a threshold.

## Quickstart

```bash
cp endpoints.yaml.example endpoints.yaml && cp .env.example .env     # edit both
python3 scripts/lib/resolve_endpoint.py --list                       # see what is configured
apptainer build --fakeroot agents/apptainer/pi/agent.sif agents/apptainer/pi/agent.def   # per agent
tests/endpoint/run.sh --endpoint default --model my-model            # does it work, what does it cost
EVAL_AGENT=pi EVAL_ENDPOINT=default EVAL_MODELS=my-model EVAL_ROUNDS=3 \
  EVAL_PROBLEMS=tiled-gemm scripts/run_eval.sh
python3 tools/summarize.py
```

Build images from the repo root (`%files` paths are relative to it). Builds need network access.

## Endpoint configuration

`endpoints.yaml` is the only place that names URLs, models and limits. Schema and an example are in
`endpoints.yaml.example`. Keys live in `.env` and are referenced by `api_key_env`.
`scripts/lib/resolve_endpoint.py` generates `.run/models.json` (pi schema, mode 0600) with every
endpoint as a provider. The agents read it:

| Agent | Protocol | Reads |
|---|---|---|
| pi | OpenAI `/chat/completions` | copies `models.json` to `~/.pi/agent/`; model must be `provider/id` |
| swe-mini | OpenAI | `.providers.$EVAL_PROVIDER.{baseUrl,apiKey}` via jq in `solve.sh` |
| kimi | OpenAI | same, exported as `KIMI_MODEL_*` |
| claude | Anthropic `/v1/messages` | same, with `/v1` stripped from the base URL |

`EVAL_ENDPOINT` selects the endpoint (optional with a single one), `EVAL_MODELS` the models by key or
wire id (default: all of the endpoint). Per-model `agent_ids` override the wire id for one agent,
`trace_tag` sets the trace folder name. Setting `MODELS_CONFIG` yourself skips the resolver and uses
your own pi-schema file. `claude-pro` ignores endpoints and uses a subscription token in
`agents/apptainer/claude-pro/.secrets/oauth_token` (see its README, never commit it).

## Environment variables

Set on the command line or in the SLURM script. `host.sh` forwards what the container needs, because
`apptainer exec` runs with `--cleanenv`.

| Variable | Default | Meaning |
|---|---|---|
| `EVAL_AGENT` | `swe-mini` | `claude`, `kimi`, `pi`, `swe-mini`, `claude-pro` |
| `EVAL_ENDPOINT`, `EVAL_PROVIDER` | only endpoint | endpoint name (`EVAL_PROVIDER` is the same value under its legacy name) |
| `EVAL_MODELS` | all models of the endpoint | comma list of keys or wire ids. Resolver output for pi is `provider/id`. |
| `EVAL_ENDPOINTS_FILE` | `endpoints.yaml` | alternative config path |
| `MODELS_CONFIG` | `.run/models.json` | own pi-schema file, bypasses the resolver |
| `EVAL_MODEL`, `EVAL_ROUND`, `EVAL_DIR` | set by host.sh / measure.sh | wire id, round number and evidence directory inside one container. Do not set them yourself. |
| `EVAL_MODEL_TAG` | model id with `/` -> `_` | trace folder name. One model only, for swapped checkpoints behind one id. |
| `EVAL_ROUNDS` | `20` | rounds per (model, problem) |
| `EVAL_ROUND_IDS` | `1..EVAL_ROUNDS` | explicit round numbers |
| `EVAL_DATE_TAG` | start time | fixed campaign directory name |
| `EVAL_PROBLEMS` | `RabbitCT-LolaBunny` | comma list of `problems/` names |
| `EVAL_SIZE`, `EVAL_VARIANT` | `256`, `LolaBunnyGPU` | RabbitCT volume size and variant name |
| `EVAL_SOLVE_PAR` | `4` | concurrent solve containers. Use 1 for CPU-timing problems. |
| `EVAL_GPU` | `auto` | `--nv` passthrough: `auto`, `1`, `0` |
| `EVAL_REQUIRE_GPU` | `0` | measure exits without writing facts when no GPU is visible (round stays unmeasured) |
| `EVAL_PERF_ONLY` | `0` | measure only the paired perf run (used by `scripts/remeasure.sh`) |
| `EVAL_NODE` | host name | recorded in `facts.json`; rounds are only comparable within a machine |
| `EVAL_MAX_TURNS` | `120` | claude and kimi iteration cap |
| `EVAL_STEP_LIMIT` | `120` | pi `--step-limit`; swe-mini `agent.step_limit` override |
| `EVAL_MAX_OUTPUT_TOKENS` | claude 64000, kimi 0 (none) | per-response output budget |
| `EVAL_MAX_THINKING_TOKENS` | unset | claude `MAX_THINKING_TOKENS`. Has no effect for models Claude Code does not recognize. |
| `EVAL_EFFORT` | `high` | `claude --effort` (a no-op for non-Anthropic models) |
| `EVAL_THINKING_EFFORT` | unset | kimi `thinking_effort` override, for servers that reject the default |
| `EVAL_KIMI_MAX_CONTEXT_SIZE` | `500000` | kimi context size, above any server window so it never compacts |
| `EVAL_KIMI_COMPACTION` | `0` | `1` restores kimi micro-compaction |
| `EVAL_CLAUDE_BLOCKING_LIMIT` | `10000000` | Claude Code client-side context block (see `docs/gotchas.md` G11) |
| `EVAL_API_TIMEOUT_MS` | `14400000` | Claude Code request timeout |
| `EVAL_MODEL_CLASS` | unset | swe-mini model class, e.g. `litellm_textbased` with `mini_textbased.yaml` |
| `MINI_CONFIG` | `agents/apptainer/swe-mini/mini.yaml` | swe-mini prompt/config file |
| `EVAL_TOOL_CHOICE_BASH` | `0` | swe-mini: pin `tool_choice` to `bash` (needed by some models) |
| `EVAL_END_GRACE_S` | `60` | pi/kimi: end a process still alive this long after its final event |
| `EVAL_SOLVE_STALL_TIMEOUT` | `3600` | kill a solve whose trace file has not changed for this long (s) |
| `EVAL_SOLVE_TIMEOUT` | `14400` | absolute per-round solve backstop (s) |
| `EVAL_SOLVE_POLL_INTERVAL` | `60` | watchdog poll interval (s) |
| `EVAL_MEASURE_TIMEOUT` | `1800` | measure phase cap (s) |
| `EVAL_SKIP_PREFLIGHT` | `0` | skip the Stage 0 probe |
| `EVAL_SKIP_CHECKS` | `0` | skip `check_pass.py` and `trace_stats.py` (caller runs them) |
| `AGENT_SIF` | `agents/apptainer/<agent>/agent.sif` | image path |
| `KIMI_BIN` | `.run/kimi-patched` | patched kimi binary, built by `make_patched_kimi.sh` |
| `KIMI_PATCH_ANY_SHA` | unset | `1` lets `make_patched_kimi.sh` patch a different kimi build |
| `DATA_ROOT` | `data/` | bulk input data root, bound read-only |
| `MEASURE_LOCK` | `$TMPDIR/agent-eval-measure.lock` | flock shared by host.sh instances on one node |
| `BENCH_ARCH` | `sm_86` | nvcc arch for the tiled-gemm bench (A40). Change it on other GPUs. |
| `OAUTH_TOKEN_FILE` | `claude-pro/.secrets/oauth_token` | claude-pro token |
| `CONTAINER_RUNTIME` | `apptainer` | only Apptainer is supported |

Per-agent notes:
- pi: stop/step cap `--step-limit`, compaction off, HTTP idle timeout 4 h in the generated `settings.json`.
- kimi: runs the patched binary: stock Kimi Code 0.31.0 with `maxCompactionPerTurn` 0 (a context overflow ends the run instead of compacting) and the OpenAI client timeout raised from 10 minutes to effectively none. `make_patched_kimi.sh` applies the two byte edits to the binary inside the image and refuses a different build.
- claude: auto-compaction off, client context block lifted, 4 h request timeout.
- swe-mini: `step_limit` and the 600 s per-command timeout come from `mini.yaml`/`solve.sh`.

## Add a problem

1. `cp -r problems/_template problems/<name>`; write `PROMPT.md` (ends with the `solution.json`
   contract), starter code that compiles and is correct but slow, a `Makefile`, and a verifier the
   prompt forbids editing.
2. `cp -r checks/_template checks/<name>`; edit `check.py`. The class is `Check(ProblemCheck)` with
   ordered `stages()` (a failing `fatal` stage skips the rest) and one non-gating `metric` stage that
   carries the ranking number. The checker lives outside `problems/<name>/` because that directory is
   copied into the agent workspace.
3. Input data: a `problems/<name>/download-input.sh` that fills `data/<name>/`, never committed.
   The agents' `host.sh` binds `data/<P>/RabbitInput` read-only for RabbitCT-style problems only. Extend
   the `DATA_BIND` block in all four `host.sh` files for another layout and keep them in sync.
4. Check the starter: `scripts/check_starter.sh <name> "<run command>"` builds it, runs it and
   passes the evidence through your checker (inside an agent image with `apptainer exec --nv` for GPU
   problems). Then run `tests/endpoint/run.sh ... --problem <name>` and update this file's problem list.

Problems: `tiled-gemm` (CUDA GEMM, graded on % of cuBLAS), `stream-compact` (CUDA scan), `2D-seq`
(CPU SOR solver, graded on wall time, `src/main.c` and the verifier are frozen and checked by an
`integrity` stage), `RabbitCT-LolaBunny` (CUDA backprojection, default) and `RabbitCT-LolaOMP`
(OpenMP). RabbitCT needs about 4.5 GB of input from `problems/RabbitCT-LolaBunny/download-input.sh`,
which `run_eval.sh` calls when `data/RabbitCT` is missing. The tiled-gemm prompt names an A40.

## Add an agent

Create `agents/apptainer/<name>/{agent.def,host.sh,solve.sh}` by copying the closest existing agent
(host.sh is near-identical across agents, keep the shared logic in sync). `solve.sh` must leave
`solution.json` and a trace file in the workspace. Teach `checks/trace_stats.py` the trace format
(`parse_round`) and `failure_class.py` its end reasons, add the agent to `tests/endpoint/check_trace.py`
(`TRACE_FILES` and the thinking extractor) and to the tables above.

## Add a model or endpoint

Add a block under `endpoints:` in `endpoints.yaml`; no code change. Run
`tests/endpoint/run.sh --endpoint <name> --model <key>` first. A server's tool-call parser must match
the model (`docs/gotchas.md` G2).

## Endpoint test

`tests/endpoint/run.sh --endpoint NAME --model KEY [--agents ...] [--steps 8] [--problem tiled-gemm] [--yes]`.
L0 (`probe.py`) sends nine protocol probes. L1 runs one `tiled-gemm` round per agent with an
8-step cap on a login node (`EVAL_GPU=0`, so the measure phase fails by design) and `check_trace.py`
asserts: trace written, stats parsed, tool calls executed and fed back, usage recorded, end reason not
an error, no parser errors, thinking present when the model is declared `reasoning: true`. The
result and the token cost are printed and saved to `tests/endpoint/last-run.md`. See
`tests/endpoint/README.md` for the measured cost.

## Serving

`serving/` is self-contained. See `serving/README.md`. It is independent of the harness and of
`endpoints.yaml`; point `base_url` at its LiteLLM port (30000).

## Results and analysis

- `python3 tools/summarize.py` prints a quick table. `python3 tools/collect_results.py` writes
  `results/runs.jsonl` + `summary.json`. Field reference: `docs/results-schema.md`.
- `reference-results/` holds the author's `summary/` (aggregates), `figures/` (example pictures),
  `analysis/` (table, HTML and figure scripts), `traces-manifest.csv` and a small `trace-example/`.
  To reproduce a figure for your own data, give a picture to your agent and ask it to rebuild it
  from `results/runs.jsonl` using `reference-results/analysis/make_figures*.py` as a reference.
  Full traces are GitHub release assets; `reference-results/README.md` says how to fetch them.
- `tools/trace-vis/`: `./run.sh -w <repo>` (build the image with `./build.sh`). Needs `results/runs.jsonl`.

## Versions the results were measured with

| Component | Version |
|---|---|
| pi | upstream `earendil-works/pi` commit `fd6659dd5d32d67feaa7ce2ba5eeb87c5705149c` (reports 0.80.2) + `agents/apptainer/pi/patches/` (`--step-limit`, 600 s bash default) |
| Kimi Code | 0.31.0 + two byte patches (`make_patched_kimi.sh`) |
| Claude Code | 2.1.209 |
| mini-swe-agent | 2.4.4 with litellm 1.91.0 |
| LiteLLM proxy (serving) | 1.89.4 + patch in `serving/litellm/patches/` |

The `agent.def` files pin these. Moving an agent to a newer version needs the trace parsers
(`checks/trace_stats.py`) and the `models.json` schema re-checked first, and the old results stop
being comparable.

## Rules for editing

- Keep verdicts on the host. Containers record facts, `checks/` judges. Do not make a check read the
  agent's self-reported result.
- Do not copy bulk data into workspaces. Do not delete `traces/`.
- Keep the four `host.sh` files in step on shared logic (workdir staging with `rsync -a`, data bind,
  trace extraction, checks, watchdog).
- Never commit `.env`, `endpoints.yaml`, `.run/`, `*.sif`, traces, or anything under `claude-pro/.secrets/`.
- Troubleshooting lives in `docs/gotchas.md`.
