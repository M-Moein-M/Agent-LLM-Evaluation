# serving

The LLM serving stack behind our evaluations: SGLang inside Apptainer on Slurm, a LiteLLM proxy
as the single front door on port 30000, Prometheus, Grafana and dcgm-exporter beside it.
Single-node jobs run one `sglang serve` or a fleet of workers behind `sglang_router`; multi-node
jobs run one TP x PP group over several nodes.

## How it works

    client (OpenAI /v1/chat/completions, Anthropic /v1/messages)
       |
       v
    LiteLLM :30000   <- litellm/config.template.yaml + reasoning_bridge.py (+ patched streaming_iterator.py)
       |
       +-- single-node, one model     -> sglang serve :BACKEND_PORT
       +-- single-node, router fleet  -> sglang_router :BACKEND_PORT -> workers :BACKEND_PORT+1..N (one per GPU group)
       +-- multi-node                 -> sglang rank 0 :BACKEND_PORT (TP x PP group over all nodes, one task per node)

    dcgm-exporter (every node) -> Prometheus :9090 <- scrapes workers, router (+LiteLLM) -> Grafana :3000

`launch.sh` reads `env.example` (and `serving/env`), writes a temp copy of `<mode>/serve.sbatch` with the
`#SBATCH` lines filled in (Slurm cannot expand variables there), and submits it. The job body sources
`common/*.sh` and `<mode>/models/<MODEL>.sh`. Clients never change: the model is only the `--served-model-name`
behind :30000 (plus `<name>-think` and a `*` catch-all on the same backend).

## Quickstart

0. This folder, `SERVE_ROOT`, `MODEL_DIR` and `IMAGE_DIR` must all be on storage that every compute node mounts.
1. Put Apptainer images in `IMAGE_DIR`: `sglang_v0.5.16.sif` (and the others a model file names),
   `dcgm-exporter-ubuntu.sif` (an ubuntu-based dcgm-exporter tag, the distroless one dies silently on
   newer drivers), `prometheus.sif`, `grafana.sif`. For example `apptainer pull prometheus.sif docker://prom/prometheus`.
2. LiteLLM venv: `python3 -m venv $SERVE_ROOT/.venv-litellm && $SERVE_ROOT/.venv-litellm/bin/pip install -r litellm/requirements.txt`,
   then `litellm/apply_patches.sh` (fixes dropped first chunks and split tool-call arguments on `/v1/messages`).
3. Models in `MODEL_DIR/<MODEL_NAME>` (the directory name is set in the model file).
4. `cp env.example env`, edit, then:

       ./launch.sh single-node                      # list models
       DRY_RUN=1 ./launch.sh single-node Qwen3.6-35B-A3B-FP8
       ./launch.sh single-node Qwen3.6-35B-A3B-FP8
       ./launch.sh multi-node Kimi-K2.7-Code-TP4xPP3 -- --dependency=afterok:123   # args after -- go to sbatch

   `DRY_RUN=1` submits and starts nothing. It prints the resolved variables, the `#SBATCH` header, every
   launch command (sglang workers, router, LiteLLM, monitoring, `srun` fan-out) and the generated `prometheus.yml`.
   `DRY_RUN_VERBOSE=1` also prints the rendered LiteLLM config.
5. When the job runs: `ssh -L 3000:localhost:3000 -L 9090:localhost:9090 <user>@<node>` (the job log prints the line),
   clients use `http://<node>:30000`. `examples/health.sh <jobid> <model-tag>` waits for a real completion.

## Variables

All are set in `env.example` with `${VAR:-default}`; a value from the shell wins over `serving/env`, which wins over the file.

| variable | default | meaning |
|---|---|---|
| `SERVE_ROOT` | `$HOME/llm-serving` | workspace on shared storage, also the job working dir |
| `MODEL_DIR` / `IMAGE_DIR` | `$SERVE_ROOT/models` / `images` | model checkouts / `.sif` files |
| `STAGE_DIR`, `STAGE_MODEL` | `/tmp/$USER-sglang-stage`, `1` | single-node: rsync the model to node-local disk (0 = serve from `MODEL_DIR`) |
| `PATCH_DIR`, `PYLIBS_DIR`, `HARMONY_CACHE` | under `SERVE_ROOT` / `MODEL_DIR` | patched sglang files, extra python libs, harmony vocab |
| `RUN_DIR`, `LOG_DIR` | `$SERVE_ROOT/runs`, `serve-logs` | per-job data (TSDB, grafana, rendered configs) / Slurm logs |
| `LITELLM_VENV`, `LITELLM_BIN` | `$SERVE_ROOT/.venv-litellm` | proxy install |
| `LITELLM_MASTER_KEY` | empty | optional auth; empty removes `general_settings` from the config |
| `LITELLM_PROMETHEUS` | `0` | `1` adds the `prometheus` callback and a `litellm` scrape job |
| `LITELLM_PORT`, `BACKEND_PORT`, `DIST_PORT` | 30000, 30060, 20000 | front door, internal sglang/router port, multi-node rendezvous |
| `DCGM_PORT`, `PROM_PORT`, `GRAFANA_PORT` | 9400, 9090, 3000 | default ports of the images |
| `ENABLE_MONITORING` | `1` | `0` skips dcgm/Prometheus/Grafana |
| `SBATCH_PARTITION`, `SBATCH_ACCOUNT`, `SBATCH_NODELIST` | `gpu`, empty, empty | empty values drop the `#SBATCH` line |
| `GPUS_PER_NODE`, `SBATCH_GRES` | 4, empty | `--gpus-per-node`, or `--gres=...` if set; multi-node size = ceil(TP x PP / GPUS_PER_NODE) |
| `CPUS_PER_TASK`, `TIME_LIMIT`, `TIME_MIN`, `SBATCH_EXTRA`, `JOB_NAME` | model default (32), `24:00:00`, empty | header values |
| `FLEET_WORKERS`, `SGLANG_WORKER_TP`, `SGLANG_ROUTER_POLICY`, ... | per model | router fleet knobs, documented in `single-node/lib.sh` |

## Add a model

Copy the closest `single-node/models/*.sh` or `multi-node/models/*.sh`, rename it to the name you will pass to
`launch.sh`, and edit. A single-node file defines `configure_model` (names, GPU count, port, image) and
`serve_sglang` (the flags, passed to `run_sglang_single` or `run_sglang_router_fleet`). A multi-node file only
defines variables (`TP_SIZE`, `PP_SIZE`, parsers, `EXTRA_ARGS`, ...): the command is built in `multi-node/lib.sh`.
Check with `DRY_RUN=1 ./launch.sh <mode> <name>` before submitting. Keep the "why" comments next to flags.

## Verify

- `DRY_RUN=1` for every model; `bash -n` on all scripts.
- `tests/start_stack.sh RUNDIR 18001 18000` starts `tests/mock_backend.py` and the real LiteLLM config on
  localhost (no GPU, `ALLOW_NO_MASTER_KEY=1` for LiteLLM 1.104.x), then
  `python3 tests/proxy_probe.py http://127.0.0.1:18000 RUNDIR LABEL` sends 15 cases on both routes and reports what
  reached the backend (reasoning echoed, thinking blocks, stream shapes). Stop with `tests/stop_stack.sh RUNDIR`.
- Against a live stack: `tests/verify_run.sh <node> <model-tag> [workers]` checks metrics, Prometheus targets and
  `cached_tokens` for a repeated prefix, directly and through LiteLLM.

## Known gotchas

- Long calls: one reply may run for the whole context window (over an hour on dense 27B models). Three limits must
  agree: LiteLLM `timeout`/`stream_timeout`/`request_timeout` 14400, and `sglang_router --request-timeout-secs 14400`.
  The router default of 1800 s cut streams at 30 min and LiteLLM reported a normal stop, so the cut looked like a
  finished reply. The router flag applies to endpoints started after the change.
- `fake_stream` must stay `false`. With `true` no byte flows until the reply is complete; Node clients (kimi, claude)
  time out after ~300 s and retry the same generation until the round limit.
- `/v1/messages` needs three config keys: `use_chat_completions_url_for_anthropic_messages: true`, a `-think` alias
  with `merge_reasoning_content_in_choices`, and `model_info.supports_reasoning: false`. The last one is not obvious:
  without it LiteLLM re-routes to `/v1/responses`, which sglang does not serve, and you get a misleading 400 about `tools.0.strict`.
- Claude Code streaming dies on turn 1 with "Content block is not a thinking block" when a reasoning model streams
  through LiteLLM's Anthropic adapter. The `-think` alias avoids it by folding reasoning into `<think>` text. Through a
  gateway you do not control, disable extended thinking in the client.
- `reasoning_bridge.py` restores `reasoning_content` on assistant turns (from `thinking_blocks`, a per-tool-call cache,
  or a placeholder). Reasoning models such as Kimi K2.x degenerate into reasoning-only loops without it.
- `litellm/patches/streaming_iterator-1.89.4.patch` targets LiteLLM 1.89.4 only. LiteLLM 1.104.x needs a master key
  (or `LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY=true`); the patch and stack were not validated on it.
- The `prometheus` LiteLLM callback and the `litellm` scrape job are untested here, and no dashboard uses
  `litellm_*` or `sgl_router_*` series. Only the sglang and dcgm dashboards are shipped.
- Do not name an exported variable `SGLANG_PORT`: sglang reads it as the base of its internal port allocation
  (the multi-node scripts use `HTTP_PORT` for this reason; `BACKEND_PORT` is the user-facing name).
- EAGLE/DSPARK speculative decoding is off in several model files after a crash when a reply reaches the context window.
  The comments in each file say how to turn it back on.
- Multi-node: leftovers of a dead job can hold the HTTP port, the script kills `sglang` processes of your user on every
  node first. HiCache models need a whole node of CPUs for host RAM. See `docs/multinode-guide.md` for fabric prerequisites.
