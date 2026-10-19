# DeepSeek-V4-Flash-0731 (4x H100, DSPARK self-speculative decoding available but off)
configure_model() {
  # Native fp8 checkpoint (block-quantized, weight_block_size 128x128), 156GB on disk.
  # TP=4 (adjusted from the vendor doc's --tp 8 for 4-GPU nodes) puts ~39GB/GPU of weights,
  # leaving ~40GB/GPU for KV cache + CUDA graphs.
  #
  # DSPARK is this checkpoint's own self-speculative decoding: config.json carries
  # dspark_target_layer_ids [40,41,42] (of 43 total layers) + dspark_block_size 5 --
  # the draft head lives INSIDE the checkpoint (like NEXTN/MTP), so no
  # --speculative-draft-model-path is needed and --speculative-dspark-block-size is left
  # to auto-infer (5) from the checkpoint. No SPS cost table / STS calibration path is
  # passed either -- omitting them just means the ragged-verify scheduler runs
  # uninitialized (verify-all), which is lossless, just without the extra throughput
  # tuning those offline-profiled tables would add.
  # DSPARK DISABLED: --speculative-algorithm DSPARK is left out of the serve line for the
  # same window-edge crash reason as the EAGLE models. Add it back (one flag, auto-infers
  # block_size 5) once the edge is tested.
  REQUIRED_NUM_GPUS=4
  MODEL_TAG="deepseek-ai/DeepSeek-V4-Flash-0731"
  MODEL_NAME="deepseek-ai/DeepSeek-V4-Flash-0731"   # dir under $MODEL_PATH (bound to /models)
  CTX_SIZE=256000
  # CTX_SIZE=131072   # smaller window for long-context-limited workloads

  SGLANG_PORT=$BACKEND_PORT   # internal; litellm front door forwards here
  SGLANG_SIF_PATH=$IMAGE_DIR/sglang_v0.5.16.sif   # confirmed: --help lists DSPARK + marlin moe-runner-backend
}

serve_sglang() {
  run_sglang_single \
    --model-path "/models/$MODEL_NAME" \
    --served-model-name "$MODEL_TAG" \
    --reasoning-parser deepseek-v4 \
    --tool-call-parser deepseekv4 \
    --moe-runner-backend marlin \
    --tp "$REQUIRED_NUM_GPUS" \
    --mem-fraction-static 0.85 \
    --enable-cache-report \
    --enable-metrics \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$SGLANG_PORT" \
    --context-length "$CTX_SIZE"
}
