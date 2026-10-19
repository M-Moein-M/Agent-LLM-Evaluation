# moonshotai/Kimi-K2.7-Code (12 GPUs = 3 nodes of 4, TP4 x PP3 + HiCache)
# Sourced by multi-node/serve.sbatch. Defines configure_model (globals) only: the srun'd
# command is built in multi-node/lib.sh from these variables. To add a model, copy a file.
configure_model() {
  # Sized for the agentic workload: ~8 concurrent sessions, mean peak context 63K, 5% of
  # rounds past 200K, ~40x more tokens read than written (mostly reused prefixes).
  #
  # Why TP4 x PP3 instead of TP16: MLA keeps a full KV copy on every TP rank, so TP16 held
  # only 605K tokens (2.3 x 256K) no matter how many GPUs. With PP each GPU stores KV only
  # for its ~20 of 61 layers -> ~1.1M tokens on 12 GPUs. TP4 keeps the per-layer
  # all-reduces on intra-node NVLink; only stage-boundary activations cross IB.
  # sglang rank order puts TP innermost, so each TP4 group is one node.
  #
  # HiCache spills evicted prefixes to host RAM (ratio 2 = ~2x the GPU pool, ~60GB/GPU,
  # pinned -> needs the memlock ulimit in multi-node/lib.sh and many CPUs for the RAM:
  # on partitions that allocate host RAM per CPU, 32 CPUs gave only ~179GB, too little),
  # so a session evicted under load reloads its prefix over PCIe instead of re-prefilling it.
  #
  # PP requires --disable-overlap-schedule and forbids speculative decoding (server_args.py).
  # CTX_SIZE=256000 (native max is 262144; the largest measured request was 255,969).
  TP_SIZE=4
  PP_SIZE=3
  NODES=$(nodes_for_gpus $((TP_SIZE * PP_SIZE)))
  RECOMMENDED_CPUS_PER_TASK=128   # whole node, for the HiCache host KV pool
  MODEL_TAG="moonshotai/Kimi-K2.7-Code"
  MODEL_NAME="moonshotai/Kimi-K2.7-Code"
  CTX_SIZE=256000
  REASONING_PARSER="kimi_k2"
  TOOL_CALL_PARSER="kimi_k2"
  HTTP_PORT=$BACKEND_PORT   # internal; litellm front door forwards here
  # 0.5.20 drops the PP bug below; on 0.5.16, re-enable EXTRA_BINDS.
  SGLANG_SIF_PATH=$IMAGE_DIR/sglang_v0.5.20.sif
  # 0.88 is safe because running requests / CUDA graphs are capped at 16 below.
  MEM_FRACTION=0.88
  EXTRA_ARGS="--disable-overlap-schedule --max-running-requests 16 --cuda-graph-max-bs-decode 16
  --enable-hierarchical-cache --hicache-ratio 2"
  # [0.5.16 only] sglang 0.5.14/0.5.16 bug: Kimi (VL wrapper) + PP crashes on PP rank >0 at CUDA-graph
  # capture ('NoneType' object has no attribute 'shape', deepseek_v2.py forward).
  # Bind a one-line-patched deepseek_v2.py over the image copy (build it with
  # sglang-patches/apply_patches.sh v0.5.16). Diff vs original:
  #   diff $SGLANG_PATCH_DIR/deepseek_v2.py.orig $SGLANG_PATCH_DIR/deepseek_v2.py
  SGLANG_PATCH_DIR=$PATCH_DIR/sglang-v0.5.16
  # Kept on 0.5.20 too (0.5.20 picks the expandable-safe v2 all-reduce, untested here).
  # No expandable_segments here: TP4 is intra-node, so sglang enables custom all-reduce, and
  # with nnodes>1 it forces the legacy v1 path, whose CUDA-IPC graph-buffer registration
  # fails on expandable-segment memory ("RuntimeError: invalid argument").
  ALLOC_CONF=""
  # EXTRA_BINDS="--bind $SGLANG_PATCH_DIR/deepseek_v2.py:/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py"
}
