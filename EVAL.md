# Evaluation workflow

## Requirements

- Docker with NVIDIA GPU runtime (CUDA 12.4)
- NVIDIA driver + `nvidia-smi`
- jq on the host (only needed if you do not set `EVAL_MODELS`)

## Architecture

The harness is split by **nature of work**:

- **Container (bash) = side effects.** Replay the agent's `build`/`run` commands with `--gpus all` and capture *raw evidence* into files. It does **not** interpret anything.
- **Host (Python) = pure judgment over files.** Parse the evidence, apply per-problem thresholds, decide pass/fail. No toolchain needed — only `python3`. Re-runnable over existing traces with zero GPU cost.

Components:

- **`scripts/run_eval.sh`** — entrypoint. Sets env vars and delegates to the host runner.
- **`agents/docker/pi/host.sh`** — builds the harness image, loops over `models x rounds x problems`, runs the container with `--gpus all`, rsyncs traces, then runs `check_pass.py` on the host.
- **`agents/docker/pi/guest.sh`** — runs the agent inside the container. The agent writes source code and a `solution.json`; the harness replays those commands and captures evidence into `.eval/` (`facts.json`, `build.log`, `run.log`, `gpu_info.txt`, `gpu_usage.csv`). No MSE parsing, no pass judgment.
- **`checks/`** — the entire judging subsystem, kept outside the agent workspace so it can't be gamed:
  - **`checks/eval_lib.py`** — shared `ProblemCheck` base class + evidence accessors (`facts()`, `log_value()`, `gpu_stats()`).
  - **`checks/<PROBLEM>/check.py`** — per-problem `Check` subclass: ordered stages + tunable threshold constants.
  - **`checks/check_pass.py`** — host-side driver: walks traces, runs each problem's `Check`, writes `.eval/pass_result.json`.
  - **`checks/collect_results.py`** — aggregates `pass_result.json` files into JSONL + a summary with pass@1 and per-stage failure counts.
- **`traces/`** — stores per-run artifacts for auditing.

### Flow

```
host.sh: loop over models/rounds/problems
  guest.sh: run agent (writes code + solution.json)
  guest.sh: replay agent's build/run, capture raw evidence → .eval/facts.json + logs + gpu_usage.csv
  host.sh: rsync traces (diff + .eval/)
host.sh → check_pass.py: judge each round → .eval/pass_result.json
collect_results.py: aggregate → runs.jsonl + summary.json
```

## Run multi-model, multi-round eval

```bash
./scripts/run_eval.sh
```

### Configuration (environment variables)

- `EVAL_MODELS`: comma-separated model IDs (default: parse from `models.json`)
- `EVAL_ROUNDS`: rounds per model (default: 20)
- `EVAL_PROBLEMS`: comma-separated problem names (default: RabbitCT)
- `EVAL_SIZE`: problem size for RabbitCT (default: 256)
- `MODELS_CONFIG`: path to `models.json` (default: `agents/apptainer/pi/models.json`)

The agent produces a `solution.json` in each problem directory with the build and run commands it used. The harness replays these commands for independent verification. No `EVAL_BUILD_CMD` / `EVAL_RUN_CMD` overrides are needed — the agent provides them.

Example:

```bash
EVAL_MODELS="deepseek-ai/DeepSeek-V4-Flash,Qwen/Qwen3.6-35B-A3B-FP8" \
EVAL_ROUNDS=20 \
./scripts/run_eval.sh
```

## Results

Each run stores logs and metrics under:

```
traces/<problem>/<model>/<timestamp>/round-<n>/
```

Files in each trace:
- `.eval/facts.json` — raw receipts: build/run commands, exit codes, timings (no interpretation)
- `.eval/build.log` — output from the build command
- `.eval/run.log` — output from the run command (includes MSE if `-c` was used)
- `.eval/gpu_info.txt` — `nvidia-smi -L` snapshot
- `.eval/gpu_usage.csv` — GPU utilization samples during run (250ms resolution)
- `.eval/pass_result.json` — the verdict: overall `passed` + per-stage results (written by `check_pass.py`)
- `changed_diff.txt` — diff of files added/changed by the agent
- `agent.jsonl` — agent session events

## Pass checks (host-side, re-runnable)

`host.sh` runs `check_pass.py` automatically after the eval loop. To re-judge
existing traces — e.g. after tuning a threshold in `checks/<PROBLEM>/check.py` —
just run it again (no GPU, no Docker):

```bash
python checks/check_pass.py --traces traces          # all problems
python checks/check_pass.py --traces traces --problems RabbitCT
```

It writes `.eval/pass_result.json` for each round. Then aggregate:

```bash
python checks/collect_results.py
```

Outputs:
- `results/runs.jsonl`
- `results/summary.json` — pass@1 per model + which stages failed (e.g. `gpu` vs `mse`)

### Tuning / adding pass criteria

- **Tune thresholds:** edit the constants at the top of `checks/<PROBLEM>/check.py`
  (e.g. `MSE_MAX`, `GPU_MIN_AVG`, `GPU_MIN_PEAK`), then re-run `check_pass.py`.
- **Add a problem's check:** create `checks/<PROBLEM>/check.py` with a `Check`
  subclass of `ProblemCheck` (see `checks/RabbitCT/check.py`). Implement `stages()`
  returning ordered stage methods; mark `build`/`run` `fatal=True` so downstream
  stages are skipped (not failed) when they break. Use the inherited `facts()`,
  `log_value(pattern)`, and `gpu_stats()` helpers — never parse raw files yourself.

## Expanding the benchmark

### Add new models

Option A: set `EVAL_MODELS` to a comma-separated list at runtime.

Option B: add models to `agents/apptainer/pi/models.json` and let the runner auto-read them.

### Add a new harness (e.g., Claude Code)

The current runner is hardwired to the Pi harness. To add another harness, create a parallel Dockerfile + guest script that match the same output contract:

- `agents/docker/<harness>/Dockerfile`
- `agents/docker/<harness>/guest.sh` (must write `.eval/facts.json` + evidence files and `agent.jsonl`)

Then extend the host runner to switch by `EVAL_HARNESS=pi|claude` and select the matching Dockerfile + guest script.

### Add new problems

Add a new folder under `problems/` with a `PROMPT.md` and any build/run assets. Then set:

```bash
EVAL_PROBLEMS="RabbitCT,MyNewProblem"
```

The agent is expected to produce a `solution.json` with `build`, `run`, and `variant_name` fields in each problem directory. Two things are needed per problem:

1. A tailored `problems/<PROBLEM>/PROMPT.md` describing the task and the required `solution.json` format.
2. A `checks/<PROBLEM>/check.py` defining the pass criteria (see "Tuning / adding pass criteria" above).

### Agent output format (solution.json)

The agent writes `solution.json` at the problem root with this structure:

```json
{
  "build": ["make TOOLCHAIN=CLANG"],
  "run": ["./run-bench.sh -s 256 -v LolaBunnyGPU -c ./RabbitInput/Reference256.vol"],
  "variant_name": "LolaBunnyGPU"
}
```

- `build` — array of commands to build the solution (replayed by the harness in order)
- `run` — array of commands to run and verify the solution (replayed by the harness in order)
- `variant_name` — the variant identifier used in `-v` flags (e.g., `LolaBunnyGPU`)

All commands must be relative to the problem root directory. If the agent fails to produce a valid `solution.json`, the harness writes a FAIL result with a reason.
