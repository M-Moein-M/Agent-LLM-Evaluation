# moonshotai/Kimi-K2.7-Code (16 GPUs = 4 nodes of 4, TP=16)
configure_model() {
  # Previous layout of the TP4xPP3 model; the plain TP16 first bring-up.
  # Native INT4 release (quantization-aware trained; the ONLY published checkpoint, so
  # INT4 is the ceiling precision -- there is no FP8/BF16 to prefer). ~600GB weights ->
  # ~37GB/GPU at TP16, the lightest fit of the guide's three models and the recommended
  # first bring-up to validate the multi-node fabric (docs/multinode-guide.md sec. 5.2). MLA
  # attention, 64 heads / TP16 = 4 heads per rank (valid on NVIDIA). Forces thinking mode.
  #
  # 256K context is supported and KV headroom (~40GB/GPU) covers it; drop CTX_SIZE to
  # 128000 if you want more concurrent requests instead of max context.
  #
  # Throughput option (add after plain TP16 is stable, not first):
  #   --dp 16 --enable-dp-attention   (put in EXTRA_ARGS)
  #
  # NOTE: the checkpoint must be on shared storage at $MODEL_PATH/moonshotai/Kimi-K2.7-Code.
  TP_SIZE=16
  PP_SIZE=1
  NODES=$(nodes_for_gpus $((TP_SIZE * PP_SIZE)))
  MODEL_TAG="moonshotai/Kimi-K2.7-Code" # Native INT4 wins on H200. Skip NVFP4 there
  MODEL_NAME="moonshotai/Kimi-K2.7-Code"
  CTX_SIZE=256000
  REASONING_PARSER="kimi_k2"
  TOOL_CALL_PARSER="kimi_k2"
  HTTP_PORT=$BACKEND_PORT   # internal; litellm front door forwards here
  SGLANG_SIF_PATH=$IMAGE_DIR/sglang_v0.5.16.sif
  MEM_FRACTION=0.85
  EXTRA_ARGS=""
}
