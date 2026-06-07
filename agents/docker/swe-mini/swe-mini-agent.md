# mini-swe-agent — notes for agent evaluation

Quick reference for using `mini-swe-agent` (v2) as a swappable agent, the way we use
`pi` / `claude` in the nhr-agent-evaluation harness.

Repo: https://github.com/SWE-agent/mini-swe-agent  ·  Docs: https://mini-swe-agent.com/latest/

---

## 1. Architecture (why it's tiny)

The whole agent is ~100 lines (`src/minisweagent/agents/default.py`). Three design choices:

- **Bash is the only tool.** No tool-calling API needed, so it runs with *any* model.
  The LM just emits shell commands as text.
- **Linear message history.** Every step appends to `self.messages`. The trajectory
  *is* exactly what gets sent to the model — great for debugging / fine-tuning.
- **Stateless execution.** Each action runs via `subprocess.run` in a fresh subshell
  (no persistent shell session). Swapping `subprocess.run` for `docker exec` is all it
  takes to sandbox — which is why it slots into any container cleanly.

### The control loop (`agents/default.py`)
```
run(task):
  messages = [system_template, instance_template(task)]   # seed
  loop:
    step()  ->  query()            # call the LM, append its message
            ->  execute_actions()  # run the bash command(s), append observation
    until a message with role == "exit" appears
  return last message's "extra" {exit_status, submission}
```
Limits are checked inside `query()`: `cost_limit` (default $3.0), `step_limit`,
`wall_time_limit_seconds`. Trajectory is re-saved after **every** step, so even a
crashed/interrupted run leaves a usable file.

### Pieces (all swappable via config)
- **agents/**: `default.py` (headless), `interactive.py` (REPL, used by the `mini` CLI by default)
- **models/**: litellm, openrouter, portkey, requesty (+ `/completion` and `/response` variants).
  Supports local models too (e.g. ollama) via litellm.
- **environments/**: `local.py`, `docker.py`, `singularity.py`, plus bubblewrap / contree / modal.
- **run/**: `mini.py` (the `mini` CLI), `hello_world.py`, `benchmarks/` (SWE-bench, ProgramBench).

### How it differs from SWE-bench batch mode
`mini-extra swebench` is SWE-bench-specific: image = **problem** (per-instance prebuilt
Docker image), agent injected, agent emits a **git patch** -> `preds.json`, and a
**separate** harness applies the patch + runs hidden tests. Our harness is the inverse:
image = **agent**, problem mounted in, agent edits in place. So we do **not** use the
swebench runner — we use plain `mini` as a local agent inside our container.

---

## 2. Install

```bash
pip install mini-swe-agent      # gives the `mini` CLI + python bindings
# or, throwaway venv:  pip install uv && uvx mini-swe-agent
```

---

## 3. pi-like behavior (non-interactive run + trajectory)

`pi` equivalent of:
```bash
pi -p --model "$EVAL_MODEL" "$PROMPT" 2>/dev/null | tee ~/agent.jsonl
```

is:
```bash
mini -y --exit-immediately \
     -m "$EVAL_MODEL" \
     -t "$PROMPT" \
     -o ~/agent.traj.json
```

Flag meanings (from `src/minisweagent/run/mini.py`):

| flag | purpose |
|------|---------|
| `-m` / `--model`      | model name (litellm style, e.g. `anthropic/claude-sonnet-4-5`, `local/llama3.1:8b`). Or set env `MSWEA_MODEL_NAME`. |
| `-t` / `--task`       | the task/prompt. Without it, mini interactively prompts. |
| `-y` / `--yolo`       | execute the model's bash actions WITHOUT per-step confirmation. |
| `--exit-immediately`  | don't prompt when the agent decides it's done. **Needed** — `-y` alone still asks before exiting. |
| `-o` / `--output`     | save the trajectory JSON here. Defaults to `<global_config_dir>/last_mini_run.traj.json` if omitted. |
| `--agent-class default` | force the bare non-interactive agent (removes all prompting machinery). Optional but cleanest for batch. |
| `-l` / `--cost-limit` | cost cap, e.g. `-l 5`. Set `0` to disable. |
| `-c` / `--config`     | config file(s) / key=value overrides (see notes). |

**Key difference vs pi:** pi streams JSON to stdout (you `tee` it). mini's stdout is a
rich/colored TUI — it writes the trajectory to a **file** via `-o` instead. Read that
file afterward. It's a single pretty-printed JSON object, **not JSONL**.

### What's in the trajectory file (`serialize()` in default.py)
```json
{
  "info": {
    "model_stats": { "instance_cost": ..., "api_calls": ... },
    "config": { "agent": {...}, "agent_type": "..." },
    "exit_status": "...",      // e.g. "Submitted", "LimitsExceeded", "TimeExceeded"
    "submission": "...",
    "mini_version": "..."
  },
  "messages": [ ... full linear history: system, task, every model turn + bash command,
                    every command output ... ],
  "trajectory_format": "mini-swe-agent-1.1"
}
```
Browse it interactively with: `mini-extra inspect ~/agent.traj.json`

### Batch over folders (problems/* style)
```bash
for dir in problems/*/; do
  ( cd "$dir" && \
    mini -y --exit-immediately \
         -m "$EVAL_MODEL" \
         -t "$(cat PROMPT.md 2>/dev/null || echo 'finish the code in main.cpp')" \
         -o "$dir/agent.traj.json" )
done
```
Give each run a distinct `-o` path — reusing one path overwrites the previous trajectory.

---

## 4. Notes on the LOCAL environment (important for our container setup)

`LocalEnvironment` (`src/minisweagent/environments/local.py`) runs each command with
`subprocess.run(shell=True, ...)` directly on the machine. It is the **default**
environment for `mini`.

- **Inside a container, use LOCAL, not docker.** When mini runs *inside* our
  apptainer/docker agent image, the container already IS the sandbox. Use the local
  environment (default). Do NOT pass `--environment-class docker` / the swebench config
  there — that would try to launch a container *inside* the container.

- **Working directory.** Commands run in `LocalEnvironmentConfig.cwd`, which defaults to
  the current working directory (`os.getcwd()`). So either `cd` into the codebase before
  launching `mini`, or set it explicitly:
  ```bash
  mini -c mini.yaml -c environment.cwd=/path/to/codebase  -m ... -t ...
  ```
  NOTE: passing any `-c` REPLACES the default config, so you must re-include `-c mini.yaml`.

- **Stateless shells.** Every action is a new subshell — `cd` / `export` do NOT persist
  between steps. The model must chain state into one command, e.g.
  `cd /problem && make && ./run`. (The default prompt already tells the model this.)

- **Per-command timeout.** Default `timeout: 30` seconds per command. CUDA builds /
  long-running binaries will hit this. Bump it:
  ```bash
  mini -c mini.yaml -c environment.timeout=600  ...
  ```

- **Env vars.** `LocalEnvironmentConfig.env` is merged over `os.environ` for each
  command. The process inherits the container's environment by default (so `EVAL_MODEL`,
  API keys, `CUDA_VISIBLE_DEVICES`, etc. set on the container are visible).

- **GPU.** mini knows nothing about GPUs — it just runs the bash the model emits. GPU
  visibility comes from how the container is launched (`apptainer exec --nv ...` /
  `docker run --gpus all ...`), not from mini.

- **Submission sentinel.** A run ends as "Submitted" when a command's output starts with
  the line `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` and exits 0; everything after that line
  becomes `submission`. (Relevant only if you use a swebench-style prompt that asks for it;
  the default `mini.yaml` prompt does not require a patch.)

---

## 5. Config quick reference

- Default config: `src/minisweagent/config/mini.yaml` (or env `MSWEA_MINI_CONFIG_PATH`).
- Override anything from the CLI with repeated `-c key=value` (recursively merged):
  ```bash
  mini -c mini.yaml \
       -c agent.step_limit=40 \
       -c agent.cost_limit=5 \
       -c environment.timeout=600 \
       -c model.model_kwargs.temperature=0 \
       -m "$EVAL_MODEL" -t "$PROMPT" -o out.traj.json
  ```
- Reminder: the moment you pass `-c`, the built-in default file is dropped — re-add
  `-c mini.yaml` first.

---

## 6. TL;DR for our harness

mini = "just another agent image" alongside pi/claude. In its `guest.sh`:
```bash
PROBLEM_DIR=$1
cd "$PROBLEM_DIR"
mini -y --exit-immediately -m "$EVAL_MODEL" \
     -t "$(cat PROMPT.md 2>/dev/null || echo 'finish the code in main.cpp')" \
     -o /output.json
```
Local environment (it's inside the container), `--nv` for GPU at the apptainer layer,
and keep grading (diff + build + GPU-utilization check) as a separate, agent-agnostic pass.
