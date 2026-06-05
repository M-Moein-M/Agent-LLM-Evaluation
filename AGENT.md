This is a description of the project and what we aim for.

## Code base structure
in `agents` folder we have the operational codes for creating sandboxes for agents with all the necessary configs and settings brought to the guest (cotainer) during benchmarking. At the moment, we are focusing on docker (easier GPU access for smaller tasks for now) and pi agent use cases . In `agents/docker/pi` we have `guest.sh` and `host.sh`. On host, we run the `host.sh` which sets ups the temp directories and calls the container (with agent installed) to solve the given problem. The `guest.sh` will run inside of the guest.

In `problems` folder we have a list of problems and codebases. Each codebase have a `PROMPT.md` file inside which determines the task that the agent has to do with that codebase. This prompt is fed to the agent in the `guest.sh`.

## Systematic comparison and Evaluation Coding agents

### Create set of programming tasks

- Describe the tasks using prompts with variable amount of detail information
- Categories:
    - Generate application from scratch
    - Perform a task within existing code
        - Refactoring
        - Optimization
        - Debugging


### Decide on LLMs to evaluate

- NHR endpoints:
- Claude models: Sonnet, Opus
- Google Gemini
- Open AI GPT 5.x
- Open Weight Models:
    - Kimi 2.6
    - Qwen
    - ...

Influence of code agent used?
- Open code
- Claude code
- Codex
- Pi
- Hermes
- Antigravity
- ??


### Hardware architectures for local deployment

- Apple Studio M1 Max
- AMD MI300X
- ??
 
### Metrics for evaluation

- How to measure the quality of the result?
- Steps necessary to get a valid result
- Reviewing the code quality
- Waiting time
- Collect experience in best practice document


## Future directions

- We might want to build something like debug-gym
    - A lightweight self-contained container image that gives the agent bug-fixing tools
    - Very helpful for the community
    - https://github.com/microsoft/debug-gym
- How to find the optimal prompt
- Mix of local and remote model
- How to optimize the quality
- How to optimize for efficient token usage



# Summary of Meeting on May 05.05.2026

#### 🔍 **1. Evaluation Scope & Hypotheses**
- Compare **same agent + different models** and **same model + different agents**
- Benchmark Chinese open models (DeepSeek, Qwen, Kimi 2.6) against Claude Opus & Sonnet
- Assess token efficiency: Claude Code appears more token-efficient than OpenCode. 
- Investigate memory management, plugin support, and context window handling across agents

#### 🛠️ **2. Tools & Infrastructure**
- Primary tools: **Claude Code**, **OpenCode**
- Use OpenCode routed through **Thomas’s endpoint**
- Create a **dedicated repo per agent** for clean isolation
- Define test cases packaged as a **tarball** (Poisson 2D) ✅ ALREADY DONE

#### 📊 **3. Benchmarking & Evaluation Framework**
- **Metrics to log:**
  - Time to First Token (TTF)
  - Total response & completion time
  - Number of steps/iterations required
  - Prompt logs & full chat history exports
  - Intermediate step commits by the model
  - Trace logs (via “Acorn” (?) or similar)
- **Benchmark tasks:**
  - Sequential CFD benchmark
  - MPI parallel implementation
  - OpenMP from scratch
  - ** Red-Black SOR solver **
  - ** CFD smoother for optimization	**
  - ** Convert sequential sparse solver → parallel with MPI **

#### 🧠 **4. Prompt Engineering & Agent Behavior Guidelines**
- **Tell the agent *what* to do, not *how***
- ⚠️ Provide incremental hints. Avoid revealing the final solution at once.
- Models degrade performance with overloaded context or multi-task prompts → **keep tasks singular & focused**
- **Claude Code config check:** May inject random context/noise → verify & lock down settings to prevent sabotage. ⚠️ Claude Code might add random stuff in the context when you are not using Claude endpoint!
- Explore **Multi-Token Prediction (MTP)** (impact on latency & quality (?))
- Determine **largest model size** feasible for deployment in Mac Studio setup

#### 🛡️ **5. Robustness & Cross-Evaluation**
- **“Evil tasks”:** Adversarial prompting to intentionally trigger bugs or hallucinations
- **Cross-model code review:** Feed one model’s output to another for evaluation & critique
- Log failure modes & recovery patterns

---
## Questions
- Do I need to use *Fritz* cluster for execution of the code?



## Sample of Output (Claude code with Qwen3.6-35B-A3B-FP8)

```text
... deciding whether it's python repo or C/C++ repo
... reading files
<think> Now I have a comprehensive understanding of the codebase. This is a C project that solves the 2D Poisson equation using the SOR (Successive Over-Relaxation) iterative method. It's part of the the cluster (National High-Performance Computing) project at University Erlangen-Nuremberg.

Let me organize my review by category and provide specific, actionable suggestions.

Key observations:

1. **Build system**: Good multi-compiler support (GCC, CLANG, ICX), but `config.mk` has a typo ("CLANG" listed twice)
2. **Memory allocation**: Uses `posix_memalign` which is not available on Windows


3. **Bug issues**: The parameter parsing macro relies on stringification with `sizeof`, but `rand` is referenced instead of `rho`, there's no validation of the `problem` parameter in `initSolver`, and a redundant function `getTimeStamp_` is defined. The `ffreeestanding` flag on GCC also conflicts with the use of standard library functions, and several minor issues like a trailing space in copyright headers and an unused `gamma` parameter need attention.

4. **Performance considerations**: The SOR solver's inner loop could be vectorized more effectively, boundary conditions are computed in separate passes, and the matrix output format lacks proper column separation.
</think>

Now I have a complete picture. Here's my review:

---

## Codebase Overview

This is a **2D Poisson equation solver** written in C, part of the the cluster (National High-Performance Computing) project at University Erlangen-Nuremberg. It uses the **SOR (Successive Over-Relaxation)** iterative method with support for multiple compilers (GCC, CLANG, ICX), optional OpenMP, LIKWID profiling, and aligned memory allocation for performance.

---

## Issues Found

### Bugs

1. **`rho` parameter silently ignored in `solve()`** ΓÇö `src/solver.c:59`
   The `Solver` struct stores `rho`, but the solve loop never uses it. The relaxation factor `factor` at line 68 is hardcoded as `solver->omega * 0.5 * (dx2 * dy2) / (dx2 + dy2)` with no reference to `rho`. This means the `rho` parameter in `poisson.par` is a no-op.

2. **`rand()` referenced but never defined** ΓÇö `src/solver.c:44`
   The `problem` parameter branches on `if (problem == 2)` and `else`, but the variable is `int problem` ΓÇö this is fine. However, there's no validation that `problem` is `1` or `2`. Passing `0` or any other value silently behaves like problem 1.

3. **`gamma` field unused** ΓÇö `src/parameter.h:12`
   The `Parameter` struct has a `gamma` field that is initialized to 0 and never read anywhere.

### Portability

4. **`posix_memalign` is Linux/macOS only** ΓÇö `src/allocate.c:16`
   `posix_memalign` is not available on Windows. This project will not compile on native Windows without a fallback (e.g., `_aligned_malloc`). The `config.mk` default is `CLANG` which is fine for cross-compilation but worth noting.

5. **`config.mk` typo** ΓÇö `config.mk:1`
   `# Supported: CLANG, CLANG, ICX` ΓÇö CLANG listed twice, GCC missing.

### Code Quality

6. **`getTimeStamp_()` is a dead function** ΓÇö `src/timing.c:22`
   This inline wrapper of `getTimeStamp()` is never called. It's likely leftover debugging cruft.

7. **`ffreeestanding` conflicts with libc use** ΓÇö `include_GCC.mk:10`
   `-ffreestanding` disables standard library support, but the code uses `stdio.h`, `stdlib.h`, `string.h`. This is a GCC-specific issue since the CLANG/ICX configs don't use this flag.

8. **`writeResult` prints `P(i,j)` with trailing space on last column** ΓÇö `src/solver.c:130`
   Every column gets `fprintf(fp, "%f ", ...)` including the last one. This leaves a trailing whitespace on each row, which can cause issues with gnuplot's matrix reading depending on the version.

9. **`#include "math.h"` with quotes** ΓÇö `src/solver.c:5`
   System headers should use `<>`, not `""`. Consistent with all other includes in this codebase, but still a style issue.

### Performance

10. **Boundary conditions in separate loops** ΓÇö `src/solver.c:91-99`
    The boundary updates for top/bottom and left/right could be merged into a single pass. Minor, but reduces loop overhead.

11. **`res` computed as per-cell mean but compared as raw energy** ΓÇö `src/solver.c:76,101`
    `res` is divided by `imax * jmax` at line 101, making it an RMS-like value. But `epssq = eps * eps` at line 71 implies the user thinks in terms of raw residual magnitude. The division by cell count makes the effective tolerance grid-dependent, which is surprising.

### Minor

12. **Copyright trailing space** ΓÇö `src/parameter.h:4`, `include_ICX.mk:1`
    Line 4 of `src/parameter.h` has `LICENSE file.    */` with trailing spaces before the comment close.

13. **`animate.plot` has hardcoded ranges** ΓÇö `animate.plot:5-6`
    `set xrange [0:40]` and `set yrange [0:40]` won't match a 300x300 grid. The visualization scripts won't work with the default `poisson.par` configuration.

---

## Summary Table

| Category | Severity | Issue |
|---|---|---|
| Bug | High | `rho` never used in SOR solver |
| Bug | Medium | `gamma` field unused in Parameter struct |
| Portability | Medium | `posix_memalign` not Windows-compatible |
| Config | Low | `config.mk` typo (CLANG listed twice) |
| Code quality | Low | Dead `getTimeStamp_()` function |
| Config | Low | `-ffreestanding` conflicts with stdlib use |
| Code quality | Low | Trailing whitespace in output format |
| Performance | Low | Boundary conditions in separate loops |
| Config | Low | Visualization ranges hardcoded to 40x40 |

The most impactful fix would be addressing the unused `rho` parameter in the solver, since it means the configuration file has a parameter that does nothing ΓÇö likely a leftover from an earlier SOR variant (possibly weighted residual correction).
```

# Evaluation Metrics

> _for next week design/propose some metrics to evaluating the quality_

🚨Keep in mind that one big advantage in out research group is compute power and multi node access. We have to at least have some sort of metric that measures the performance.

⚠️ *We should note that these metrics are heavily task dependent. Moreover, metrics can demostrate different things when are used for one-shot tasks vs. agentic tasks.*

For example, in [AgentBench](https://arxiv.org/pdf/2308.03688), they said " averaged score is heavily impacted by tasks that generally yield higher scores" and they normalize the scores 
 - How can we merge all the tasks into one single value?
 - What to adjust to improve the metric for agentic workflows.

### Pass@k (from HumanEval)

Estimate the probability that at least one of the top k samples is correct among n generated samples. In other words, if we pick k items from the pool of n generated solutions, what is the probability of this solution being correct. Pass@1 for example, tells you the probability that the solution is correct the first time.

### Expected Token

It is defined as:
$$E(\kappa) = T_{\text{avg}} / \text{pass@1}$$

For example, if you have 10% probability of getting the solution right at the first time and it costs you 1000 tokens on average, the $E(\kappa)$ would be $1000/10\% = 10000$ .

💡One improvement can be to have a figure for the context length to see how it grows. The shape shows whether the context is ballooning, increasing linearly. Where would the first big jump be? Would assume now that it's at the start.

### [MT@4](https://arxiv.org/abs/2605.24110v1) (Multi-Turn Persistent Score)

Breaking down the whole tasks in to multiple steps and also investigate the failure rate and recovery rate.
1. implementation & compile & execution
2. verification against gold results
3. Scalability (properly parallelized)

More granular metric compared to **pass@1**. For example for each failed step, we can reset the trace and ask the model again to attempt to finish the task plus the given error message

This helps us to find out where does the agent fails.

We can also calculate **token usage** for each stage.

### Runtime 

One thing to consider is [Kaplan-Meier](https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator) calculation for the average runtime. If the computation exceeds the max timeout, we should not consider the runtime as timeout. Read more in the introduction section of [this paper](https://arxiv.org/pdf/2406.06647).

### Strong/weak scaling
How the generated code behaves as node count or problem size grows (HPC-specific, not in general benchmarks)

*Weak scaling* measures how the solution time stays constant when you increase both the problem size and the number of processors proportionally (fixed time and want to solve a bigger problem).

*Strong scaling* measures how the solution time decreases when you add more processors to a fixed problem size (fixed problem and want it faster).



### Roofline proximity
How close to theoretical hardware limits the generated code gets? 

The challenge will be in detecting the main kernel and calculating the operations. Maybe evaluating agents' capabilities in calculating the theoritical performance limit can be a topic for another project.

### Energy Efficiency ([related paper](https://arxiv.org/pdf/2601.19311))
The paper looks at batch size and serving efficiencies of (small and mid-size)LLM models using frameworks like *vllm*. But our work might benefit more from measuring energy consumption of other frameworks like *llama.cpp* which is mostly used for serving one user. ⚠️ However, it should be noted that focusing on energy, takes away the attention from the quality and scalability of the code, the thing that is less focused in literature.

---

### Use the advantage we have!💪

**Scaling efficiency (strong + weak)** — impossible to measure without multi-node or multi-GPU access. You need actual MPI runs at p = 1, 2, 4, 8, 16, 32 to plot the curve. Simulators can't fake this.

**Roofline proximity** — requires hardware-specific profiling (Nsight, ROCm profiler, perf) against the actual peak bandwidth and FLOPs of your specific devices. The ceiling is meaningless without knowing the machine.

**Kaplan-Meier runtime** — requires running code to completion (or timeout) on real hardware under realistic load. Wall-clock timing on shared cluster infrastructure, with proper statistical treatment of censored runs, is only credible if you actually have the cluster.

**speedup@k** — needs a real serial baseline on the same hardware. Cross-machine comparisons are junk; your advantage is that baseline and generated code run on identical hardware under identical conditions.


The others — pass@k, MT@4, E(κ), context growth, iterations to pass — are hardware-agnostic and could in principle be run anywhere. They're still valuable, but they're not your differentiator.

---


# HPC Dwarfs

There is already some HPC Dwarfs applications implemented by the cluster [here](https://github.com/HPC-Dwarfs).

The goal is to design a list of buggy code snippets based on HPC Dwarfs where edit-only agents have harder time to tackle. 
Some inspiration can be taken from [debug-gym](https://github.com/microsoft/debug-gym/blob/main/data/mini_nightmare/mini_nightmare.md) repo. 
> This is a selection of human generated buggy code that give conventional code repairing system some mini panic attack.

## Bug Characteristics

- It is Semantically correct, contextually wrong. The Code compiles, runs, and often produces almost correct output. 

- No syntax errors, typos, or obviously suspicious patterns

- They require a mental model outside the code itself (thread scheduling, physics convention)

- The bug and its symptom are far apart. In Molecular Dynamics for example, a corrupted cell index propagates through neighbor lists → force calculations → particle trajectories

- The fix is a single character or line, but the reasoning to get there is long

➡️One idea could be looking at the commit messages and detect the changes that once were fixed but 

## 1. MD-Bench
[Postponed for later]

## 2. RabbitCT

Due to the availability of validation checks in this repo, we focused on this.
