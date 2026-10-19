# Qwen3.6-27B (FLEET_WORKERS x TP=2, default 2 workers on 4 GPUs)
configure_model() {
  # Qwen3.5 hybrid (linear/full attention) multimodal model with a built-in MTP head
  # (mtp_num_hidden_layers=1), so EAGLE spec decode runs off the model's own MTP --
  # no separate draft model.
  # tp 2 / dp 2, not tp 1 / dp 4: the bf16 weights are 51.7GiB, so at tp 1 a 0.8
  # static fraction leaves ~11.5GiB per GPU, and the 16 full-attention layers
  # (4 kv heads x 256 head_dim, bf16) cost 64KiB/token, so CTX_SIZE 256000 needs
  # 15.6GiB of KV plus the mamba state cache. The pool would land near 115-185K and
  # replies would stop short of the window. tp 2 halves the weights and shards the
  # KV, leaving room for the full window several times over.
  # --mamba-scheduler-strategy extra_buffer manages the linear-attention state cache.
  # TOPOLOGY: FLEET_WORKERS (default 2) independent TP=2 `sglang serve` workers (2 GPUs each,
  # SGLANG_WORKER_TP=2) behind `sglang_router --policy cache_aware` on :$SGLANG_PORT
  # (run_sglang_router_fleet). Replaces `sglang serve --dp 2` (round_robin: agent turns
  # scattered across the two replicas and re-prefilled). REQUIRED_NUM_GPUS = FLEET_WORKERS * 2.
  FLEET_WORKERS=${FLEET_WORKERS:-2}
  SGLANG_WORKER_TP=2
  REQUIRED_NUM_GPUS=$((FLEET_WORKERS * SGLANG_WORKER_TP))
  MODEL_TAG="Qwen/Qwen3.6-27B"
  MODEL_NAME="Qwen3.6-27B"
  CTX_SIZE=256000
  SGLANG_PORT=$BACKEND_PORT   # internal; litellm front door forwards here
  SGLANG_SIF_PATH=$IMAGE_DIR/sglang_v0.5.16.sif
  declare_sglang_fleet "$FLEET_WORKERS"   # two TP=2 workers behind the cache-aware router
}

serve_sglang() {
  # Speculative decoding (EAGLE/MTP) DISABLED: SGLang 0.5.14 crashes mid-decode in
  # the spec-v2 + Mamba verify path -- set_mamba_track_indices_from_reqs hits a None
  # cache slot for a running req and torch.tensor() raises 'NoneType' cannot be
  # interpreted as an integer. Re-enable the SGLANG_ENABLE_SPEC_V2 export + the four
  # --speculative-* flags once an upstream fix lands.
  # export_env SGLANG_ENABLE_SPEC_V2 1
  # The cu130 0.5.14 image ships a broken openai dep -- it's missing the 'distro'
  # module and crashes on import. Inject a pip-installed copy from $PYLIBS_DIR onto
  # PYTHONPATH so the serve import chain loads. (pylibs created with:
  #   apptainer exec $SGLANG_SIF_PATH pip install --target $PYLIBS_DIR distro )
  PYLIBS=$PYLIBS_DIR
  SGLANG_APPTAINER_OPTS=(--env "PYTHONPATH=$PYLIBS")
  run_sglang_router_fleet \
    --model-path "/models/$MODEL_NAME" \
    --served-model-name "$MODEL_TAG" \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --mamba-scheduler-strategy extra_buffer \
    --mem-fraction-static 0.8 \
    --enable-cache-report \
    --enable-metrics \
    --trust-remote-code \
    --context-length "$CTX_SIZE"
}
