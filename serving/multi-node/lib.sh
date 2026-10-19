#!/bin/bash
# Multi-node helpers: one SGLang process per node, together one TP x PP group. Needs
# common/lib.sh. The batch script runs serve_sglang (cleanup + rendezvous + srun fan-out);
# every srun task runs node_main on its own node.
#
# Models in multi-node/models are too large for a single node and need TP (and PP) across
# nodes. One task per node; each task runs one SGLang process that owns its node's GPUs.
# NODES = ceil(TP_SIZE * PP_SIZE / GPUS_PER_NODE).
#
# Weights are loaded directly from the shared MODEL_DIR (visible on every compute node), so
# there is no node-local staging here. Staging a ~600GB checkpoint to local disk on each
# node every run would copy several TB and risk filling it; the shared load is simpler and
# only happens once at startup. To use local NVMe, rsync the model into a node-local dir
# with `srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 rsync ...` and repoint MODEL_PATH.

# The srun'd script that kills leftovers. The [s] bracket is load-bearing: pkill -f matches
# full command lines, and this very bash -c contains the pattern text. "[s]glang" matches
# "sglang" in orphan processes but not the literal "[s]glang" in our own cmdline -- without
# it the cleanup task kills itself and srun aborts the whole job before sglang ever starts.
CLEANUP_SCRIPT='
    pkill -u "$USER" -f "[s]glang" 2>/dev/null
    fuser -k -n tcp "$HTTP_PORT" 2>/dev/null
    true
  '

serve_sglang() {
  # Blocks that don't use pipeline parallelism or extra flags leave these unset.
  PP_SIZE=${PP_SIZE:-1}
  EXTRA_ARGS=${EXTRA_ARGS:-}
  EXTRA_BINDS=${EXTRA_BINDS:-}

  # Rendezvous address = node-rank-0's IP. node 0 is the batch node (this script runs
  # here), which is also where litellm runs, so the SGLang HTTP API (bound on rank 0)
  # is reachable at localhost:$HTTP_PORT for the proxy. We use the management/host IP
  # for the rendezvous; the NCCL *data* path still goes over IB (auto-detected, or pin
  # it via the NCCL_* knobs below).
  local node0_host node0_ip dist_init_addr
  node0_host=$(list_nodes | head -n1)
  if is_dry; then
    node0_ip="<node0-ip>"
    dist_init_addr="${node0_ip}:${DIST_PORT}"
  else
    # Prefer IPv4 for the rendezvous: getent hosts can return an IPv6 address, and a bare
    # IPv6 + ":port" is ambiguous (SGLang's NetworkAddress.parse rejects it, wanting
    # [ipv6]:port). This is only the TCP rendezvous addr; the NCCL data path still uses IB.
    node0_ip=$(getent ahostsv4 "$node0_host" | awk '{print $1; exit}')
    if [ -n "$node0_ip" ]; then
      dist_init_addr="${node0_ip}:${DIST_PORT}"
    else
      # No IPv4; fall back to IPv6 and bracket it as SGLang requires.
      node0_ip=$(getent ahostsv6 "$node0_host" | awk '{print $1; exit}')
      dist_init_addr="[${node0_ip}]:${DIST_PORT}"
    fi
  fi
  log "Multi-node SGLang: nnodes=$SLURM_NNODES tp=$TP_SIZE pp=$PP_SIZE dist-init=$dist_init_addr"

  # Nodes are unpinned and a previous run's sglang can survive job teardown still holding
  # $HTTP_PORT. Node 0's HTTP server then fails to bind ([Errno 98] address already in
  # use), warmup times out against the stale listener, and the whole TP group collapses
  # with gloo "Connection closed by peer". Kill leftovers on every node first.
  export HTTP_PORT
  run srun --overlap --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c "$CLEANUP_SCRIPT"

  # Export everything the per-node srun tasks need. srun propagates this env to every
  # task; $SLURM_NODEID is set by SLURM *per task* (0 = batch node), so node_main reads it
  # on each node.
  export MODEL_PATH MODEL_NAME SGLANG_SIF_PATH MODEL_TAG CTX_SIZE HTTP_PORT TP_SIZE
  export REASONING_PARSER TOOL_CALL_PARSER MEM_FRACTION PP_SIZE EXTRA_ARGS EXTRA_BINDS
  export DIST_INIT_ADDR="$dist_init_addr"
  # Forward fragmentation tuning into the container (APPTAINERENV_ prefix).
  # Models can opt out with ALLOC_CONF="" (see the Kimi TP4 x PP3 model).
  ALLOC_CONF=${ALLOC_CONF-expandable_segments:True}
  if [ -n "$ALLOC_CONF" ]; then
    export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF
  fi
  # If NCCL doesn't auto-pick IB, pin the interface/HCA (names vary per cluster), e.g.:
  #   export APPTAINERENV_NCCL_SOCKET_IFNAME=ib0
  #   export APPTAINERENV_NCCL_IB_HCA=mlx5
  export APPTAINERENV_NCCL_DEBUG=INFO
  # --overlap: the dcgm-exporter step (start_monitoring) already holds a step on
  # every node; without it this step could wait on CPUs that step claimed.
  if is_dry; then
    log "(dry run) per node, srun would execute node_main; showing the command for node-rank 0:"
    run srun --overlap --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
      bash -c '. "$SERVING_DIR/common/lib.sh"; . "$SERVING_DIR/multi-node/lib.sh"; node_main'
    node_main
  else
    srun --overlap --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
      bash -c '. "$SERVING_DIR/common/lib.sh"; . "$SERVING_DIR/multi-node/lib.sh"; node_main'
  fi
}

# Runs on every node, one srun task each.
node_main() {
  local extra_binds extra_args cmd host
  host=$(node_hostname)
  # Do NOT --bind /dev/infiniband here. Apptainer already exposes the host /dev
  # (devtmpfs) wholesale, IB devices included. An explicit user bind overlays it
  # with a nodev-flagged mount, so every open() of a uverbs node fails EACCES --
  # NCCL then logs "NET/IB: Unable to open device mlx5_X" and silently falls back
  # to NET/Socket, crippling cross-node TP.
  #
  # memlock: ibverbs pins registration memory; keep the rlimit unlimited.
  run ulimit -l unlimited 2>/dev/null || echo "[$host] WARNING: cannot raise memlock ulimit (current: $(ulimit -l))"

  # Default TRITON_CACHE_DIR is $HOME/.triton/cache, which on clusters with a shared home
  # is visible to every node. All TP workers across all nodes then JIT-compile into the
  # SAME cache path; on a cold or mismatched cache (e.g. after switching image -> different
  # triton build) they compile concurrently and race on the same .so, so one worker's
  # import hits a file another worker is mid-replacing ("cannot open shared object file:
  # No such file or directory"), which SIGQUITs the whole TP group. Fix: give each node a
  # private, node-local cache dir so no two nodes ever touch the same compile target.
  # Job-ID-scoped so a stale dir from a dead job cannot collide with this run.
  export APPTAINERENV_TRITON_CACHE_DIR="/tmp/triton-cache-${SLURM_JOB_ID}-${host}"
  run mkdir -p "$APPTAINERENV_TRITON_CACHE_DIR"

  echo "[$host] node-rank=$SLURM_NODEID starting sglang"
  # EXTRA_BINDS / EXTRA_ARGS are space/newline separated words, as in the model files.
  set -f
  # shellcheck disable=SC2206
  extra_binds=($EXTRA_BINDS)
  # shellcheck disable=SC2206
  extra_args=($EXTRA_ARGS)
  set +f
  cmd=(apptainer exec --nv
    --bind "$MODEL_PATH:/models"
    ${extra_binds[@]+"${extra_binds[@]}"}
    "$SGLANG_SIF_PATH" sglang serve
    --model-path "/models/$MODEL_NAME"
    --served-model-name "$MODEL_TAG"
    --tool-call-parser "$TOOL_CALL_PARSER"
    --reasoning-parser "$REASONING_PARSER"
    --tp "$TP_SIZE"
    --pp-size "$PP_SIZE"
    --nnodes "$SLURM_NNODES"
    --node-rank "$SLURM_NODEID"
    --dist-init-addr "$DIST_INIT_ADDR"
    --mem-fraction-static "$MEM_FRACTION"
    --trust-remote-code
    --host 0.0.0.0
    --port "$HTTP_PORT"
    --enable-metrics
    --enable-mfu-metrics
    --context-length "$CTX_SIZE"
    ${extra_args[@]+"${extra_args[@]}"})
  run "${cmd[@]}"
}
