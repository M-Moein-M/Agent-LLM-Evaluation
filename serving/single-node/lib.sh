#!/bin/bash
# Single-node helpers: model staging and the two SGLang topologies. Needs common/lib.sh.
#
# ============================================================================
# SGLANG LAUNCH HELPERS -- two topologies, called from a model's serve_sglang():
#
#   run_sglang_single ARGS...
#     One `sglang serve` in the foreground. Use for DP<=1 (a single radix tree,
#     so repeats always hit the warm cache -- no router needed). Caller passes the
#     full arg list; nothing topology-specific is added. Prometheus scrapes
#     $SGLANG_PORT directly (SGLANG_METRICS_PORTS defaults to it).
#
#   declare_sglang_fleet N
#     Call in configure_model AFTER setting SGLANG_PORT. Sets NUM_WORKERS=N and the
#     two metrics-port vars start_monitoring reads (engine /metrics move off
#     $SGLANG_PORT -- that becomes the router -- onto $SGLANG_PORT+1..+N; the
#     router's own sgl_router_* metrics land on $SGLANG_PORT+100).
#
#   run_sglang_router_fleet ARGS...
#     N independent `sglang serve` (one per GPU via CUDA_VISIBLE_DEVICES, or all on
#     SGLANG_WORKER_GPU if set), each TP=$SGLANG_WORKER_TP (default 1) on
#     127.0.0.1:$SGLANG_PORT+i, fronted by sglang_router on $SGLANG_PORT. --policy
#     cache_aware (default) hashes each request's prefix and pins it to the worker
#     already holding it -- fixes the --dp round-robin scatter across cold per-replica
#     trees. The router exec always gets --bind /models + --model-path so cache_aware
#     can load the tokenizer (without it the router 404s fetching it from HuggingFace).
#     Caller's ARGS are the per-worker `sglang serve` args MINUS --tp/--host/--port
#     (the helper appends those).
#
# Optional env knobs (set before the run_* call):
#   SGLANG_ROUTER_POLICY    routing policy (default cache_aware; round_robin for
#                           identical replicas where the tokenizer hash is wasted)
#   SGLANG_EVICTION_SECS    router tree-eviction interval (default 240)
#   SGLANG_CACHE_THRESHOLD  min prefix-match ratio for cache routing (default 0.5; router
#                           default 0.3 let the shared system prompt pull every session
#                           onto one worker)
#   SGLANG_BALANCE_ABS/_REL load gap that switches to shortest-queue routing (default 4 /
#                           1.2; router default 64 / 1.5 never triggers at eval concurrency)
#   WORKER_STARTUP_TIMEOUT  router's wait for workers to come healthy (default 2400)
#   SGLANG_WORKER_GPU       pin every worker to this one GPU id (default: i-th GPU);
#                           when set, workers start one at a time (each waits for /health)
#   SGLANG_WORKER_TP        GPUs per worker (default 1; worker i gets GPUs (i-1)*TP .. i*TP-1,
#                           so N workers need N*TP GPUs)
#   SGLANG_APPTAINER_OPTS   bash array of extra apptainer options placed after the model bind,
#                           for example (--env PYTHONPATH=/x --bind /a:/b:ro)
# ============================================================================

# Node-local staging dir, FIXED (not the per-job random $TMPDIR, which SLURM wipes at
# job end). If the job is pinned to one node (SBATCH_NODELIST), a re-run lands on the
# same node and rsync sees the model already present, so the multi-hundred-GB copy
# becomes a cheap metadata scan. rsync is also crash-safe: it stages each file under a
# temp name and renames on completion, so an interrupted run resumes correctly (no
# partial file is ever loaded). Assumes node-local /tmp survives between jobs.
stage_model() {
  # Stages $MODEL_NAME plus, if the model set it, $DRAFT_MODEL_NAME (speculative-decoding
  # draft). Both must land under the SAME stage root, since MODEL_PATH is repointed there
  # and the container sees only that one bind.
  local name src dest
  if [ "${STAGE_MODEL:-1}" != 1 ]; then
    log "staging disabled (STAGE_MODEL=$STAGE_MODEL), serving from $MODEL_PATH"
    return 0
  fi
  for name in "$MODEL_NAME" ${DRAFT_MODEL_NAME:+"$DRAFT_MODEL_NAME"}; do
    src="$MODEL_PATH/$name"
    dest="$STAGE_DIR/$name"
    log "Syncing model from $src to $dest (rsync; near no-op if already current) ..."
    run mkdir -p "$dest"
    run rsync -a --info=progress2 "$src/" "$dest/"
    log "Model ready at $dest"
    # pre-warm the page cache: the loader mmaps the shards and faults them in with
    # tiny ~12KB reads (latency-bound, ~375MB/s); streaming them sequentially into
    # RAM first lets the NVMe stripe run at full speed and the load become RAM-fast
    log "Pre-warming page cache ..."
    if is_dry; then
      echo "+ find $(shq "$dest") -name '*.safetensors' -print0 | xargs -0 -P 16 -I{} cat {} > /dev/null"
    else
      find "$dest" -name '*.safetensors' -print0 |
        xargs -0 -P 16 -I{} cat {} >/dev/null
    fi
    log "Cache warm"
  done
  MODEL_PATH="$STAGE_DIR"
}

run_sglang_single() {
  local cmd=(apptainer exec --nv
    --bind "$MODEL_PATH:/models"
    ${SGLANG_APPTAINER_OPTS[@]+"${SGLANG_APPTAINER_OPTS[@]}"}
    "$SGLANG_SIF_PATH" sglang serve "$@")
  run "${cmd[@]}"
}

declare_sglang_fleet() {
  NUM_WORKERS=$1
  # engine /metrics: workers on $SGLANG_PORT+1 .. +NUM_WORKERS (prometheus scrapes
  # these, NOT $SGLANG_PORT -- that is the router). The official SGLang dashboard's
  # PromQL has no instance= filter, so these sum into node-wide totals for free.
  SGLANG_METRICS_PORTS=$(seq -s' ' $((SGLANG_PORT + 1)) $((SGLANG_PORT + NUM_WORKERS)))
  # router's own sgl_router_* series (per-worker routing decisions, queue depth)
  SGLANG_ROUTER_METRICS_PORT=$((SGLANG_PORT + 100))
}

run_sglang_router_fleet() {
  local base_port=$((SGLANG_PORT + 1))
  local worker_args=("$@")
  local urls=() i rport cmd
  SGLANG_WORKER_PIDS=()
  local tp=${SGLANG_WORKER_TP:-1} gpus
  for i in $(seq 1 "$NUM_WORKERS"); do
    rport=$((base_port + i - 1))
    # worker i owns GPUs (i-1)*tp .. i*tp-1 (a single id when tp=1)
    gpus=${SGLANG_WORKER_GPU:-$(seq -s, $(((i - 1) * tp)) $((i * tp - 1)))}
    log "launching $MODEL_TAG worker $i/$NUM_WORKERS (tp=$tp) on 127.0.0.1:$rport (GPU $gpus) ..."
    # apptainer --env splits its value on commas, so a multi-GPU list goes in via
    # APPTAINERENV_ instead; tp=1 keeps the original --env path.
    if [ "$tp" -eq 1 ]; then
      cmd=(apptainer exec --nv
        --bind "$MODEL_PATH:/models"
        --env "CUDA_VISIBLE_DEVICES=$gpus"
        ${SGLANG_APPTAINER_OPTS[@]+"${SGLANG_APPTAINER_OPTS[@]}"}
        "$SGLANG_SIF_PATH" sglang serve
        "${worker_args[@]}"
        --tp 1
        --host 127.0.0.1
        --port "$rport")
    else
      cmd=(env "APPTAINERENV_CUDA_VISIBLE_DEVICES=$gpus"
        apptainer exec --nv
        --bind "$MODEL_PATH:/models"
        ${SGLANG_APPTAINER_OPTS[@]+"${SGLANG_APPTAINER_OPTS[@]}"}
        "$SGLANG_SIF_PATH" sglang serve
        "${worker_args[@]}"
        --tp "$tp"
        --host 127.0.0.1
        --port "$rport")
    fi
    run_bg "${cmd[@]}"
    SGLANG_WORKER_PIDS+=("$LAST_BG_PID")
    urls+=("http://127.0.0.1:$rport")
    # Workers sharing one GPU (SGLANG_WORKER_GPU) must start one at a time: sglang sizes
    # its KV pool from free memory measured at its own startup, so a worker that measures
    # while a sibling is still allocating gets a negative budget and dies
    # ("Loaded weights leave no GPU memory for the KV cache").
    if [ -n "${SGLANG_WORKER_GPU:-}" ] && [ "$i" -lt "$NUM_WORKERS" ]; then
      if is_dry; then
        log "(dry run) would wait for http://127.0.0.1:$rport/health before starting the next worker"
      else
        local waited=0
        until curl -sf -o /dev/null "http://127.0.0.1:$rport/health"; do
          if ! kill -0 "$LAST_BG_PID" 2>/dev/null; then echo "WARNING: worker $i exited during startup"; break; fi
          if [ "$waited" -ge "${WORKER_STARTUP_TIMEOUT:-2400}" ]; then echo "WARNING: worker $i not healthy after ${waited}s"; break; fi
          sleep 5; waited=$((waited + 5))
        done
        echo "worker $i startup wait done after ${waited}s, starting next"
      fi
    fi
  done
  log "starting sglang_router (${SGLANG_ROUTER_POLICY:-cache_aware}) on :$SGLANG_PORT over ${urls[*]}"
  cmd=(apptainer exec
    --bind "$MODEL_PATH:/models"
    "$SGLANG_SIF_PATH" python -m sglang_router.launch_router
    --worker-urls "${urls[@]}"
    --policy "${SGLANG_ROUTER_POLICY:-cache_aware}"
    --eviction-interval-secs "${SGLANG_EVICTION_SECS:-240}"
    --cache-threshold "${SGLANG_CACHE_THRESHOLD:-0.5}"
    --balance-abs-threshold "${SGLANG_BALANCE_ABS:-4}"
    --balance-rel-threshold "${SGLANG_BALANCE_REL:-1.2}"
    --model-path "/models/$MODEL_NAME"
    --worker-startup-timeout-secs "${WORKER_STARTUP_TIMEOUT:-2400}"
    # 14400 s: the router default of 1800 s silently cut every stream at 30 min, and the
    # cut looks like a normal stop. Must stay >= the LiteLLM timeouts in the config.
    --request-timeout-secs 14400
    --prometheus-host 127.0.0.1
    --prometheus-port "${SGLANG_ROUTER_METRICS_PORT:-$((SGLANG_PORT + 100))}"
    --host 0.0.0.0
    --port "$SGLANG_PORT")
  run "${cmd[@]}"
}
