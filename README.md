# Systematic comparison and Evaluation Coding agents

A evaluation framework for benchmarking LLMs and coding agents on High Performance Computing (HPC) tasks — measuring correctness, runtime performance, and agentic iteration behavior on real parallel and distributed workloads.

---

## Motivation

Existing benchmarks like ParEval test static code generation for parallel programming, but fall short in three key areas:
- they don't evaluate iterative agent workflows
- they lack real performance measurement at scale
- they omit modern execution models like SYCL and Fortran

This project aims to address all three gaps with access to real multi-GPU, multi-node cluster hardware.

---

## Goals

- Evaluate both **raw LLMs** (API/non-agentic) and **coding agents** (Claude Code, OpenHands, Codex CLI) on HPC tasks
- Measure **correctness**, **runtime performance**, and **agentic loop behavior** in a single unified framework
- Produce a **public benchmark** that others can reproduce and build on
- Create a leaderboard that updates quarterly and updates the results on a dedicated webpage.
---

## Architecture

The benchmark is organized into three layers:

### Layer 1 — Task Library (based on ParEval)

A structured collection of HPC problems across execution models and task types.

**Execution models covered:**
- MPI (distributed memory)
- CUDA (NVIDIA GPU)
- OpenMP (shared memory + GPU offloading)
- HIP / ROCm (AMD GPU)
- SYCL / oneAPI
- Kokkos (portability layer)
- Fortran (legacy modernization)

**Task types:**
- **Refactoring**
- **Optimization**
- **Debugging**
- Generation — write parallel code from scratch given a specification
- Translation — convert serial code to a parallel model
- Migration — port between execution models (e.g., CUDA → OpenMP offloading, CUDA → SYCL)
- Optimization — improve performance of existing parallel code guided by profiling output


**Difficulty tiers:**
- Tier 1: Single function / kernel
- Tier 2: Multi-file module
- Tier 3: Full mini-application

**Problem domains** include: stencil codes, FFTs, sparse solvers, sorting, histograms, molecular dynamics patterns, and CFD kernels.

---

### Layer 2 — Evaluation Engine

Runs on real multi-GPU / multi-node cluster hardware. Measures three dimensions per task:

**Correctness**
- `pass@k` against reference unit tests
- Compile success rate
- Runtime correctness (output validation against reference)

**Performance**
- `speedup@k` and `efficiency@k` relative to serial baseline
- Strong scaling (fixed problem size, increasing node count)
- Weak scaling (problem size grows with node count)
- Roofline analysis (how close to theoretical bandwidth / compute limits)
- Memory bandwidth utilization

**Agentic loop**
- Number of iterations to reach a passing solution
- Compile-failure recovery rate
- Runtime-failure recovery rate
- `E(κ)` — expected token cost to reach a passing solution

---

### Layer 3 — Agent Harness

A sandboxed HPC environment that wraps real CLI agents and exposes them to benchmark tasks through a controlled loop.

**LLMs to evaluate:**
- NHR endpoints:
- Claude models: Sonnet, Opus
- Google Gemini
- Open AI GPT 5.x
- Open Weight Models:
    - Kimi 2.6
    - Qwen
    - ...

**Supported targets:**
- Open code
- Pi
- Hermes
- Antigravity
- Claude Code (Anthropic agentic CLI)
- Codex CLI (OpenAI agent)
- OpenHands (open-source agent framework)
- Raw LLMs via API (non-agentic baseline)

**Environment requirements per agent run:**
- Full compiler stack: GCC, NVCC / HIPcc / clang+SYCL
- MPI runtime (`mpirun`, `srun`)
- GPU passthrough or SLURM multi-node access
- Controlled wall-clock and token budgets per task
- Isolated containers with resource limits

---

## Current implementation

Layers 2 and 3 above are the target design. What is built today is a working
pass@1 harness for GPU kernel problems (RabbitCT first), organized around one
principle: **separate side effects from judgment.**

- **Container (bash) = side effects.** The agent solves the problem inside a
  GPU container, then the harness *replays the agent's own build/run commands*
  and captures **raw evidence** to files. It interprets nothing.
- **Host (Python) = judgment over files.** A separate, re-runnable step parses
  that evidence, applies per-problem thresholds, and decides pass/fail. It needs
  only `python3` — no compiler, no GPU — so re-tuning a threshold recomputes
  pass@1 over existing traces at zero GPU cost.

This decoupling means the expensive GPU run happens once, while the verdict can
be re-derived cheaply and audited independently of what the agent self-reported.

### Pipeline

```
host.sh        loop over models × rounds × problems; run container with --gpus all
  guest.sh     run agent → it writes source + solution.json (its build/run cmds)
  guest.sh     replay build/run; capture raw evidence → .eval/
  host.sh      rsync trace (agent diff + .eval/)
host.sh →
  check_pass.py  judge each round against checks/<PROBLEM>/check.py → pass_result.json
collect_results.py  aggregate → pass@1 per model + per-stage failure breakdown
```

### Components

| Path | Role |
|---|---|
| `agents/docker/pi/host.sh` | Orchestrator: builds the image, loops, runs containers, rsyncs traces, invokes the host-side checks. |
| `agents/docker/pi/guest.sh` | In-container: runs the agent, replays its `solution.json` commands, captures evidence. **No interpretation.** |
| `problems/<P>/PROMPT.md` | Task description + the `solution.json` contract the agent must produce. |
| `checks/` | Self-contained judging subsystem, kept outside the agent workspace so it can't be gamed. |
| `checks/eval_lib.py` | Shared `ProblemCheck` base + evidence accessors (`facts()`, `log_value()`, `gpu_stats()`). |
| `checks/<P>/check.py` | Per-problem pass criteria: ordered stages + tunable threshold constants. |
| `checks/check_pass.py` | Host-side driver: runs each problem's check over a trace → `pass_result.json`. |
| `checks/collect_results.py` | Aggregates verdicts → pass@1 per model and which stages fail. |

### Evidence captured per round (`.eval/`)

- `facts.json` — raw receipts: build/run commands, exit codes, timings (no MSE, no verdict)
- `build.log`, `run.log` — untouched command output
- `gpu_info.txt` — `nvidia-smi -L` device identity
- `gpu_usage.csv` — GPU utilization sampled at 250 ms during the run
- `pass_result.json` — the verdict: overall `passed` + per-stage results (written host-side)

### Pass criteria per problem

Each problem defines a `checks/<P>/check.py` with a `Check` subclass of
`ProblemCheck`. Stages run in order; `build`/`run` are *fatal* (if they fail,
downstream stages are `skipped`, not failed — preserving "which stage broke").
Thresholds are constants at the top of the file, so re-running `check_pass.py`
after editing one re-judges existing traces with no GPU re-run. RabbitCT checks
`build → run → mse → gpu` (MSE under threshold against the reference volume, and
GPU utilization above a floor to catch CPU fallback).

> Operational details — running the eval, configuration env vars, and adding new
> problems — are in [EVAL.md](EVAL.md).

---

## Hardware architectures for local deployment
- Apple Studio M1 Max
- AMD MI300X
- ??

## Metrics for evaluation

- How to measure the quality of the result?
- Steps necessary to get a valid result
- Reviewing the code quality
- Waiting time
- Collect experience in best practice document


## Relationship to Existing Work

| Project | What it contributes | Gap this project fills |
|---|---|---|
| ParEval | 420 problems, 7 execution models, correctness + basic speedup metrics | No agent harness, no real-scale performance, no SYCL/Fortran |
| ParEval-Repo | Repo-level translation tasks, `E(κ)` cost metric | Very early, limited execution model coverage |
| ParaCodex | Agent workflow prototype for OpenMP offloading | Single execution model, not a general benchmark |
| HPC-Coder-V2 | Fine-tuned model for HPC tasks | Training data project, not an eval framework |
| HeCBench | Hardware performance benchmarks | Not LLM-focused |

This project extends ParEval's task library and drivers where useful, but builds an independent evaluation and agent harness layer.

---

## Known Risks and Open Questions

- **Reproducibility:** HPC timing on shared infrastructure is noisy. Statistical rigor (multiple runs, variance reporting, node isolation) is required.
- **Agent sandboxing:** Giving real agents access to real cluster resources is a security and cost challenge. A clean isolation layer is needed before this can be opened to third parties.
- **Scope creep:** Covering all execution models at launch is risky. The plan is to phase: nail MPI + CUDA depth first, then add SYCL, HIP, and Fortran in subsequent releases.
- **LLM coverage of newer models:** SYCL and Fortran are underrepresented in training data — signal may be sparse at first.
- **Benchmark saturation:** With better coding agents and LLMs, the performance of all will plateu and there would not be a meaningful difference between the models.

---

## Status

Early design phase. Immediate next steps:
1. Define task library schema (problem format, metadata, difficulty tiers)
2. Design agent harness runner and sandbox architecture
3. Establish baseline results on ParEval problems to validate infrastructure
4. Extend task set with scientific domain kernels

## Future directions

- How to find the optimal prompt
- Mix of local and remote model
- How to optimize the quality
- How to optimize for efficient token usage
- How access to library docs (OpenMP, ...) affect the performance

---

## Coding Agents

Here's a focused comparison table of the main CLI coding agents, filtered for your preference — **opinionated, fixed workflows** over plugin-heavy customization:

| Agent | Local LLM (OpenAI-compat) | Workflow Philosophy | Customizability |
|---|---|---|---|
| **Claude Code** | ✅ (via endpoint override) | Most opinionated. Autonomous loops: plan → edit → test → fix. Fixed agentic workflow with hooks and Agent Teams. | Low. Hooks and CLAUDE.md for context, but workflow is largely fixed by Anthropic. |
| **Aider** | ✅ (via LiteLLM) | Pair-programmer style. Git-centric: every change auto-commits. Architect mode uses two models (planner + editor). | Low-medium. Fixed modes (architect, editor, ask, help). You configure which models fill roles, not the workflow. |
| **Goose** | ✅ (OpenAI-compatible) | Autonomous, system-level. Complex multi-step task loops. Co-designed by Block + Anthropic on MCP. | Medium. MCP-extensible, but core workflow (plan → act → verify) is fixed. |
| **OpenCode** | ✅ (LOCAL_ENDPOINT env var) | Dual-agent built-in: `build` (full access) vs `plan` (read-only). LSP integration baked in. | Medium-high. Custom agents via markdown files, 75+ providers. More flexible than the others on this list. |
| **Gemini CLI** | ❌ (Gemini models only) | Opinionated around Google's stack. File ops, shell exec, web search baked in. 1M token context. | Low. Fixed Google toolchain. Least configurable. |
| **Codex CLI** | ❌ (OpenAI models only) | Sandboxed execution, careful permissions model. Runs code in isolated env by default. | Low. Locked to OpenAI, no MCP, minimal config. |
| **Plandex** | ✅ (OpenAI-compatible) | Long-horizon planning focus. 2M token context, diff sandbox before applying changes. Self-hosted server option. | Low-medium. Multi-model routing built in, but structured workflow is its selling point. |

**Based on your criteria** (robust + fixed workflows, not plugin-driven), the strongest picks are:

- **Claude Code** — most mature opinionated workflow, best SWE-bench scores, least wiggle room by design
- **Aider** — battle-tested, the architect/editor split is a genuinely fixed workflow that's hard to break, excellent git integration
- **Plandex** — underrated, the diff sandbox means changes are staged before touching your files, very structured


## How to setup Claude code with custom model

Claude code can be installed based on the docs: https://use-code.com/index

Afterwards, a file in `~/.claude/settings.json` should be created with the following content. The settings values should be configured according to available models, API key, and endpoint.

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

Detailed reading is available at:
- https://jonathansblog.co.uk/using-claude-code-with-local-llm-models-the-complete-guide
- https://code.claude.com/docs/en/model-config
- https://use-code.com/index

## How to setup Codex with custom model (NOT COMPLETE)

Check the official repo for the latest release: https://github.com/openai/codex/

On Linux, look for `codex-x86_64-unknown-linux-musl.tar.gz` and unzip it. It will give you a binary file that you can move to `~/.local/bin/` folder so you can access it via terminal (given that the `~/.local/bin/` is added to your PATH environment variable).

⚠️ Codex works with `/response` endpoint now. The `/chat` does not function:
```bash
Error loading config.toml: `wire_api = "chat"` is no longer supported.
How to fix: set `wire_api = "responses"` in your provider config.
```