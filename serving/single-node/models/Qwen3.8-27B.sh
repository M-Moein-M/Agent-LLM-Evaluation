# Qwen3.8-27B (4x H100, 2 workers x TP=2 + cache-aware router)
configure_model() {
  # Same qwen3_5 hybrid (linear/full attention) family as Qwen3.6-27B, with its own
  # built-in MTP head (mtp_num_hidden_layers=1) -- EAGLE spec decode runs off that,
  # no separate draft model. bf16 weights are ~52GB on disk, fit on one 80GB H100,
  # but TP=1 leaves only ~165K fp8 KV tokens per worker (< the 256K context) and
  # queued 10-12 requests at 0.86 KV usage, so run 2 workers x TP=2 instead. Native
  # max_position_embeddings is 262144, so the requested 256K context needs no YaRN.
  #
  # TOPOLOGY: NOT `sglang serve --dp 2` (one process, sglang's internal DP balancer).
  # Instead the run_sglang_router_fleet helper -- 2 independent `sglang serve` (TP=2,
  # GPUs 0-1 and 2-3), each on its own localhost port, with `sglang_router --policy
  # cache_aware` in front on $SGLANG_PORT. cache_aware pins a repeated long prefix to
  # the worker that already holds it instead of scattering across cold per-replica trees.
  # See common helpers header (single-node/lib.sh) for the knobs; nothing here overrides the defaults.
  #
  # KNOWN RISK: Qwen3.6-27B (same architecture family, same node) had to
  # DISABLE EAGLE/MTP spec decode on sglang 0.5.14 -- it crashes mid-decode in the
  # spec-v2 + Mamba verify path (set_mamba_track_indices_from_reqs hits a None cache
  # slot). Spec decode is off here too (see serve_sglang).
  REQUIRED_NUM_GPUS=4
  MODEL_TAG="Qwen/Qwen3.8-27B"
  MODEL_NAME="Qwen/Qwen3.8-27B"   # dir under $MODEL_PATH (bound to /models) -- nested under Qwen/
  CTX_SIZE=256000
  SGLANG_PORT=$BACKEND_PORT   # router front door; litellm forwards here
  SGLANG_SIF_PATH=$IMAGE_DIR/sglang_qwen38-27b.sif   # purpose-built image for this model
  SGLANG_WORKER_TP=2
  declare_sglang_fleet 2   # two TP=2 workers behind the cache-aware router
}

serve_sglang() {
  # Speculative decoding DISABLED: same window-edge crash risk as the other EAGLE models --
  # illegal memory access in process_batch_result_decode when a reply reaches the context
  # window. To re-enable, put these four back on the serve line:
  #   --speculative-algorithm EAGLE --speculative-num-steps 3
  #   --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  run_sglang_router_fleet \
    --model-path "/models/$MODEL_NAME" \
    --served-model-name "$MODEL_TAG" \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --attention-backend flashinfer \
    --chunked-prefill-size 8192 \
    --max-prefill-tokens 16384 \
    --max-running-requests 8 \
    --mamba-full-memory-ratio 4.21 \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-ssm-dtype bfloat16 \
    --enable-cache-report \
    --enable-metrics \
    --trust-remote-code \
    --context-length "$CTX_SIZE"
}
