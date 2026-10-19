# google/gemma-4-31B-it (FLEET_WORKERS x TP=SGLANG_WORKER_TP, default 2 x 2 on 4 GPUs)
configure_model() {
  # Dense 31B, bf16 weights ~62GB: FLEET_WORKERS (default 2) TP=SGLANG_WORKER_TP (default 2) workers behind
  # `sglang_router --policy cache_aware` on :$SGLANG_PORT (run_sglang_router_fleet). Replaces
  # `sglang serve --dp 4` (round_robin: every turn of an agent session hit a different
  # replica and re-prefilled). GPUs needed = FLEET_WORKERS * SGLANG_WORKER_TP (default 4).
  # KV (smoke test): bf16 KV at TP=1 gives ~27.8K tokens/worker, at TP=2 ~136K, both below
  # the 256K window. Default is therefore 2 TP2 workers (cache_aware across the two) with
  # --kv-cache-dtype fp8_e4m3, which roughly doubles the TP2 pool. Measure the
  # "KV Cache is allocated. #tokens" line against CTX_SIZE after any change.
  FLEET_WORKERS=${FLEET_WORKERS:-2}
  SGLANG_WORKER_TP=${SGLANG_WORKER_TP:-2}
  REQUIRED_NUM_GPUS=$((FLEET_WORKERS * SGLANG_WORKER_TP))
  MODEL_TAG="google/gemma-4-31B-it"
  MODEL_NAME="google/gemma-4-31B-it"   # dir under $MODEL_PATH (bound to /models)
  CTX_SIZE=256000
  SGLANG_PORT=$BACKEND_PORT   # router front door; litellm forwards here
  SGLANG_SIF_PATH=$IMAGE_DIR/sglang_v0.5.16.sif
  declare_sglang_fleet "$FLEET_WORKERS"   # TP=$SGLANG_WORKER_TP workers behind the cache-aware router
}

serve_sglang() {
  run_sglang_router_fleet \
    --model-path "/models/$MODEL_NAME" \
    --served-model-name "$MODEL_TAG" \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4 \
    --mem-fraction-static 0.85 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-cache-report \
    --enable-metrics \
    --trust-remote-code \
    --context-length "$CTX_SIZE"
}
