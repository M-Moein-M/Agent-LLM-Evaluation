# Qwen3.6-35B-A3B-FP8 (FLEET_WORKERS x 1 GPU, default 4)
configure_model() {
  # TOPOLOGY: FLEET_WORKERS (default 4) independent TP=1 `sglang serve` workers, one per GPU,
  # behind `sglang_router --policy cache_aware` on :$SGLANG_PORT (run_sglang_router_fleet).
  # Replaces `sglang serve --dp 4`, whose internal balancer is round_robin: every turn of an
  # agentic session landed on a different replica, so the prefix cache mostly missed.
  # cache_aware pins a session's prefix to the worker that already holds it.
  # FP8 weights (~35GB) fit on one GPU. Smoke test on 2 GPUs: FLEET_WORKERS=2 GPUS_PER_NODE=2 launch.sh ...
  # (REQUIRED_NUM_GPUS is only informational here, it must equal FLEET_WORKERS).
  FLEET_WORKERS=${FLEET_WORKERS:-4}
  REQUIRED_NUM_GPUS=$FLEET_WORKERS
  MODEL_TAG="Qwen/Qwen3.6-35B-A3B-FP8"
  MODEL_NAME="Qwen3.6-35B-A3B"
  CTX_SIZE=256000
  SGLANG_PORT=$BACKEND_PORT   # router front door; litellm on :$LITELLM_PORT forwards here
  SGLANG_SIF_PATH=$IMAGE_DIR/sglang_v0.5.16.sif
  declare_sglang_fleet "$FLEET_WORKERS"   # one TP=1 worker per GPU behind the cache-aware router
}

serve_sglang() {
  export_env SGLANG_ENABLE_SPEC_V2 1
  run_sglang_router_fleet \
    --model-path "/models/$MODEL_NAME" \
    --served-model-name "$MODEL_TAG" \
    --reasoning-parser qwen3 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mamba-scheduler-strategy extra_buffer \
    --tool-call-parser qwen3_coder \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.88 \
    --disable-overlap-schedule \
    --enable-cache-report \
    --enable-metrics \
    --trust-remote-code \
    --context-length "$CTX_SIZE"
}
