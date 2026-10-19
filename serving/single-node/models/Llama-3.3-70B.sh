# Llama-3.3-70B-Instruct (2 GPUs, TP=2)
configure_model() {
  # Dense 70B bf16 weights are ~141GB total -- at TP=1 that is the whole capacity of one
  # 141GB H200 card with no room for KV cache. TP=2 puts ~70.6GB/GPU of weights, leaving
  # ~70GB/GPU for KV cache + CUDA graphs. Raise DP_SIZE to run replicas (needs
  # REQUIRED_NUM_GPUS = TP_SIZE * DP_SIZE GPUs).
  REQUIRED_NUM_GPUS=2
  TP_SIZE=2
  DP_SIZE=1
  MODEL_TAG="meta-llama/Llama-3.3-70B-Instruct"
  MODEL_NAME="Llama-3.3-70B-Instruct"
  CTX_SIZE=130000 # max ctx length of llama3.3
  SGLANG_PORT=$BACKEND_PORT   # internal; litellm front door forwards here
  SGLANG_SIF_PATH=$IMAGE_DIR/sglang_v0.5.16.sif
}

serve_sglang() {
  # The cu130 0.5.14 image ships a broken openai dep -- it's missing the 'distro'
  # module and crashes on import. Inject a pip-installed copy from $PYLIBS_DIR onto
  # PYTHONPATH so the serve import chain loads. (pylibs created with:
  #   apptainer exec $SGLANG_SIF_PATH pip install --target $PYLIBS_DIR distro )
  PYLIBS=$PYLIBS_DIR
  SGLANG_APPTAINER_OPTS=(--env "PYTHONPATH=$PYLIBS")
  run_sglang_single \
    --model-path "/models/$MODEL_NAME" \
    --served-model-name "$MODEL_TAG" \
    --tool-call-parser llama3 \
    --mem-fraction-static 0.85 \
    --tp "$TP_SIZE" \
    --dp "$DP_SIZE" \
    --enable-cache-report \
    --enable-metrics \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$SGLANG_PORT" \
    --context-length "$CTX_SIZE"
}
