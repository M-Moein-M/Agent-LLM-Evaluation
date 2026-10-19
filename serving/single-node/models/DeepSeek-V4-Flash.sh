# DeepSeek-V4-Flash (4 GPUs, TP=4)
configure_model() {
  REQUIRED_NUM_GPUS=4
  MODEL_TAG="DeepSeek-V4-Flash"
  MODEL_NAME="DeepSeek-V4-Flash"
  CTX_SIZE=256000
  SGLANG_PORT=$BACKEND_PORT   # internal; litellm front door forwards here
  SGLANG_SIF_PATH=$IMAGE_DIR/sglang_v0.5.16.sif
}

serve_sglang() {
  # Speculative decoding DISABLED: Qwen3.6-35B crashed with an illegal memory access in
  # process_batch_result_decode when a reply hit the context window, under EAGLE + the
  # overlap scheduler. Same combination here, and this model has never been driven to the
  # window. To re-enable, restore the export below plus these four on the serve line:
  #   --speculative-algorithm EAGLE --speculative-num-steps 3
  #   --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  # export_env SGLANG_ENABLE_SPEC_V2 1
  run_sglang_single \
    --model-path "/models/$MODEL_NAME" \
    --served-model-name "$MODEL_TAG" \
    --reasoning-parser deepseek-v4 \
    --tool-call-parser deepseekv4 \
    --moe-runner-backend marlin \
    --tp "$REQUIRED_NUM_GPUS" \
    --enable-cache-report \
    --enable-metrics \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$SGLANG_PORT" \
    --context-length "$CTX_SIZE"
}
