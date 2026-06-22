# Systematic Comparison and Evaluation of Coding Agents

An evaluation framework for benchmarking LLMs and coding agents on High
Performance Computing (HPC) tasks — measuring **correctness**, **runtime
performance**, and **agentic iteration behavior** on real parallel and
distributed workloads.

> **Who this README is for:** human readers — to understand what the project
> is, why it exists, how it's built, and how to run it. If you are an AI agent
> working *in* this repo, read [`AGENT.md`](AGENT.md) instead — it has the
> operational details (exact run flow, what not to touch, where output lands,
> and the environment gotchas).

---

## Motivation

Existing benchmarks like ParEval test static code generation for parallel
programming, but fall short in three key areas:

- they don't evaluate iterative **agent** workflows,
- they lack real **performance** measurement at scale,
- they omit modern execution models like SYCL and Fortran.

This project addresses all three gaps, with access to real multi-GPU,
multi-node cluster hardware as the differentiator.

## Goals

- Evaluate both **raw LLMs** (API / non-agentic) and **coding agents** (Claude
  Code, OpenHands, Codex CLI, pi, …) on HPC tasks.
- Measure **correctness**, **runtime performance**, and **agentic loop
  behavior** in a single unified framework.
- Produce a **public, reproducible benchmark** with a leaderboard that updates
  quarterly on a dedicated webpage.

---

## What's built today

The long-term design is a three-layer system (see [Target
architecture](#target-architecture) below). What runs **today** is a working
**pass@1 harness** for GPU/CPU kernel problems and a full mini-application
(RabbitCT), with **multiple swappable agents** (`pi` and `swe-mini`) run over
the same problems on real hardware.

The whole thing is organized around one principle: **separate side effects from
judgment.**

- **Container (bash) = side effects.** The selected agent solves the problem
  inside a container, then the harness *replays the agent's own build/run
  commands* and captures **raw evidence** to files. It interprets nothing.
- **Host (Python) = judgment over files.** A separate, re-runnable step parses
  that evidence, applies per-problem thresholds, and decides pass/fail. It needs
  only `python3` — no compiler, no GPU — so re-tuning a threshold recomputes
  pass@1 over existing traces at zero GPU cost.

Because judgment is agent-agnostic (it reads `.eval/` evidence, not the agent),
adding a new agent is just a new container + invocation — the checks are reused.

### High-level architecture

```
scripts/run_eval.sh              single entrypoint
  ├─ Stage 1  ensure bulk input data is present (e.g. RabbitCT's ~4.5 GB input)
  └─ Stage 2  exec agents/docker/$EVAL_AGENT/host.sh
                ├─ solve   agent writes source + solution.json (its build/run cmds)
                ├─ measure replay build/run with GPU sampling → raw evidence in .eval/
                └─ extract trace → traces/<problem>/<agent>/<model>/<ts>/round-N/
checks/check_pass.py             judge each round vs checks/<problem>/check.py
checks/collect_results.py        aggregate → pass@1 per (agent, model)
```

The two currently implemented agents:

| Agent      | Runtime                                  | Notes |
|------------|------------------------------------------|-------|
| `pi`       | `@mariozechner/pi-coding-agent` (Node)   | Two-phase: parallel *solve* (shared GPU) + serial *measure* (GPU-isolated). |
| `swe-mini` | `mini-swe-agent` (Python, litellm)       | Two-phase like pi: parallel *solve* (shared GPU) + serial *measure*. Bash-tool agent talking to an OpenAI-compatible gateway. |

Both read the same gateway config and produce the same evidence shape, so the
checks judge them identically.

---

## Getting started

### Prerequisites

- **Docker** with NVIDIA GPU passthrough (`docker run --gpus all`). On Windows
  this means **Docker Desktop with WSL2 integration**; the pipeline must be run
  from inside WSL (see [`AGENT.md`](AGENT.md) for the exact invocation and why).
- An **NVIDIA GPU** + recent driver (verified locally on a GTX 1050).
- **`python3`** on the host (for the pass/fail checks — no GPU needed).
- **`jq`** on the host (only if you don't set `EVAL_MODELS` explicitly; used to
  read model IDs from the config).
- Access to an **OpenAI-compatible LLM gateway** (the NHR endpoint). Its URL +
  API key live in `agents/apptainer/pi/models.json`.

### Run it

```bash
./scripts/run_eval.sh
```

That runs the default agent (`pi`) over the default problem
(`RabbitCT-LolaBunny`) for every model in the config. Common overrides:

```bash
EVAL_AGENT=swe-mini \
EVAL_MODELS="deepseek-ai/DeepSeek-V4-Flash,Qwen/Qwen3.6-35B-A3B-FP8" \
EVAL_ROUNDS=20 \
./scripts/run_eval.sh
```

Results land in `traces/` (per-round evidence) and `results/summary.json`
(pass@1 per model). The full list of environment variables, the single-problem
workflow, and re-judging existing traces without a GPU are documented in
[`AGENT.md`](AGENT.md).

---

## Target architecture

The benchmark is designed as three layers. Layer 1 and parts of Layers 2–3 are
in progress; the rest is the roadmap.

### Layer 1 — Task Library (based on ParEval)

A structured collection of HPC problems across execution models and task types.

- **Execution models:** MPI, CUDA, OpenMP (incl. GPU offloading), HIP/ROCm,
  SYCL/oneAPI, Kokkos, Fortran.
- **Task types:** generation, translation (serial → parallel), migration
  (CUDA → OpenMP/SYCL), optimization (profiling-guided), refactoring, debugging.
- **Difficulty tiers:** single kernel → multi-file module → full mini-application.
- **Domains:** stencil codes, FFTs, sparse solvers, sorting, histograms,
  molecular dynamics, CFD kernels.

### Layer 2 — Evaluation Engine

Runs on real multi-GPU / multi-node hardware. Measures three dimensions:

- **Correctness:** `pass@k`, compile success rate, runtime output validation.
- **Performance:** `speedup@k` / `efficiency@k` vs serial baseline, strong &
  weak scaling, roofline proximity, memory-bandwidth utilization.
- **Agentic loop:** iterations to a passing solution, compile/runtime-failure
  recovery rate, `E(κ)` (expected token cost to pass).

### Layer 3 — Agent Harness

A sandboxed HPC environment that wraps real CLI agents and exposes them to
benchmark tasks through a controlled loop.

- **LLMs to evaluate:** Claude (Sonnet, Opus), Gemini, GPT 5.x, and open-weight
  models (Kimi 2.6, Qwen, DeepSeek, …) via NHR endpoints.
- **Agent targets:** Claude Code, Codex CLI, OpenHands, OpenCode, pi, Hermes,
  Antigravity, and raw-LLM-via-API baselines.
- **Environment per run:** full compiler stack (GCC, NVCC/HIPcc/clang+SYCL),
  MPI runtime, GPU/SLURM access, controlled wall-clock + token budgets,
  isolated containers.

---

## Evaluation metrics

> These metrics are heavily **task-dependent**, and they mean different things
> for one-shot vs agentic tasks. A key open question is how to merge many tasks
> into a single comparable value (cf. AgentBench's note that averaged scores are
> dominated by easy, high-scoring tasks).

**Our differentiator is compute** — multi-node / multi-GPU access. The metrics
that *require* real hardware are the ones we should lead with:

- **Strong & weak scaling** — needs actual MPI runs at p = 1, 2, 4, 8, 16, 32.
  Simulators can't fake this.
- **Roofline proximity** — needs hardware-specific profiling (Nsight, ROCm
  profiler) against the real peak bandwidth/FLOPs of the device.
- **Kaplan-Meier runtime** — running to completion (or timeout) on real
  hardware, with proper statistical treatment of censored (timed-out) runs.
- **speedup@k** — needs a real serial baseline on the *same* hardware;
  cross-machine comparisons are junk.

Hardware-agnostic metrics (still valuable, but not the differentiator):

- **pass@k** (HumanEval) — probability that at least one of `k` samples is
  correct. pass@1 is the probability the first attempt is right.
- **E(κ)** — expected token cost to pass: `E(κ) = T_avg / pass@1`. (e.g. 10%
  pass@1 at 1000 tokens average ⇒ 10 000.) A companion *context-growth* plot
  shows whether context balloons or grows linearly, and where the first jump is.
- **MT@4** (multi-turn persistent score) — break a task into stages
  (implement+compile+run → verify vs gold → scalability), measure per-stage
  failure and recovery rate, and token usage per stage. More granular than
  pass@1: shows *where* an agent fails. On a failed stage you can reset the
  trace and re-ask with the error message.
- **Energy efficiency** — measuring serving energy (e.g. `llama.cpp` for
  single-user serving). Caveat: focusing on energy pulls attention away from
  code quality and scalability, which are under-studied.

---

## Relationship to existing work

| Project | What it contributes | Gap this project fills |
|---|---|---|
| ParEval | 420 problems, 7 execution models, correctness + basic speedup | No agent harness, no real-scale performance, no SYCL/Fortran |
| ParEval-Repo | Repo-level translation tasks, `E(κ)` cost metric | Very early, limited execution-model coverage |
| ParaCodex | Agent workflow prototype for OpenMP offloading | Single execution model, not a general benchmark |
| HPC-Coder-V2 | Fine-tuned model for HPC tasks | A training-data project, not an eval framework |
| HeCBench | Hardware performance benchmarks | Not LLM-focused |

This project extends ParEval's task library and drivers where useful, but builds
an independent evaluation and agent-harness layer on top.

### "Nightmare" debugging problems

A parallel track (inspired by [debug-gym's
mini-nightmares](https://github.com/microsoft/debug-gym/blob/main/data/mini_nightmare/mini_nightmare.md))
builds buggy HPC code that **edit-only agents struggle with** — bugs that are
*semantically* subtle rather than syntactic:

- compiles, runs, and produces *almost* correct output;
- no syntax errors or obviously suspicious patterns;
- requires a mental model outside the code (thread scheduling, physics
  convention);
- bug and symptom are far apart (e.g. a corrupted cell index in molecular
  dynamics propagating through neighbor lists → forces → trajectories);
- the fix is one line, but the reasoning to get there is long.

Seeds are drawn from the the cluster [HPC-Dwarfs](https://github.com/HPC-Dwarfs)
applications. RabbitCT is the first focus because it ships validation checks;
MD-Bench is queued. See `problems/NIGHTMARES.md` for the catalog.

---

## Known risks and open questions

- **Reproducibility:** HPC timing on shared infrastructure is noisy; statistical
  rigor (multiple runs, variance reporting, node isolation) is required.
- **Agent sandboxing:** giving real agents real cluster resources is a security
  and cost challenge; a clean isolation layer is needed before opening to third
  parties.
- **Scope creep:** covering all execution models at launch is risky — the plan
  is to nail MPI + CUDA depth first, then add SYCL, HIP, Fortran.
- **Sparse signal for new models:** SYCL and Fortran are underrepresented in
  training data.
- **Benchmark saturation:** as agents/LLMs improve, scores may plateau and lose
  discriminating power.

## Status & next steps

Early phase. Immediate next steps:

1. Define the task-library schema (problem format, metadata, difficulty tiers).
2. Harden the agent-harness runner and sandbox.
3. Establish baseline results on ParEval problems to validate the infrastructure.
4. Extend the task set with scientific-domain kernels.

**Future directions:** finding optimal prompts, mixing local + remote models,
optimizing for quality and token efficiency, and measuring how access to library
docs (OpenMP, …) affects performance.

---

## Appendix: coding agents compared

Filtered for a preference toward **opinionated, fixed workflows** over
plugin-heavy customization:

| Agent | Local LLM (OpenAI-compat) | Workflow philosophy | Customizability |
|---|---|---|---|
| **Claude Code** | ✅ (endpoint override) | Most opinionated. Autonomous plan → edit → test → fix, hooks, Agent Teams. | Low — workflow largely fixed by Anthropic. |
| **Aider** | ✅ (LiteLLM) | Pair-programmer, git-centric (auto-commits). Architect mode = planner + editor. | Low–medium — fixed modes, you pick which models fill roles. |
| **Goose** | ✅ | Autonomous, system-level, multi-step loops. MCP-based (Block + Anthropic). | Medium — MCP-extensible, core loop fixed. |
| **OpenCode** | ✅ (`LOCAL_ENDPOINT`) | Dual-agent: `build` (full access) vs `plan` (read-only). LSP baked in. | Medium–high — custom agents via markdown, 75+ providers. |
| **Gemini CLI** | ❌ (Gemini only) | Opinionated around Google's stack. 1M context. | Low. |
| **Codex CLI** | ❌ (OpenAI only) | Sandboxed execution, careful permissions. | Low — no MCP, minimal config. |
| **Plandex** | ✅ | Long-horizon planning, 2M context, diff sandbox before applying. | Low–medium — multi-model routing built in. |

Strongest picks for robust + fixed workflows: **Claude Code** (most mature
opinionated loop), **Aider** (battle-tested architect/editor split), **Plandex**
(diff sandbox stages changes before touching files).

### Using Claude Code with a custom model

Install per the [docs](https://code.claude.com/docs/en/model-config), then
create `~/.claude/settings.json`:

```json
{
  "env": {
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "ANTHROPIC_MODEL": "Qwen/Qwen3.6-35B-A3B-FP8",
    "ANTHROPIC_BASE_URL": "https://gateway.example.org/api/llmgw/",
    "ANTHROPIC_AUTH_TOKEN": "sk-REDACTED..."
  },
  "theme": "dark"
}
```

Further reading:
[model-config docs](https://code.claude.com/docs/en/model-config) ·
[local-LLM guide](https://jonathansblog.co.uk/using-claude-code-with-local-llm-models-the-complete-guide).

> ⚠️ When *not* using a real Claude endpoint, Claude Code may inject extra
> context/noise — verify and lock down settings.

### Using Codex CLI with a custom model (incomplete)

Grab the latest release from the [Codex repo](https://github.com/openai/codex/)
(`codex-x86_64-unknown-linux-musl.tar.gz` on Linux), unpack the binary into a
`PATH` directory (e.g. `~/.local/bin/`). Codex now requires the `/responses`
endpoint — `wire_api = "chat"` is no longer supported, so set
`wire_api = "responses"` in the provider config.
