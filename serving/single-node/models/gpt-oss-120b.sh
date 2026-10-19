# openai/gpt-oss-120b (2x H100, TP=2) with local parser and chat-template patches
configure_model() {
  # MXFP4 MoE checkpoint, 61GB of safetensors, served with TP=2 on two H100 (95.8GiB
  # each). After load 86.0GiB per card is in use (weights plus a KV pool of 1.1M tokens).
  # Requires 2 visible GPUs (GPUS_PER_NODE=2 at submit).
  # No --moe-runner-backend / --attention-backend on purpose: sglang's
  # GptOssForCausalLM override picks triton_kernel (the MXFP4 MoE path) and fa3 on
  # sm_90 by itself, and forces dtype bf16.
  # The model answers in harmony channels (analysis / commentary / final).
  # --reasoning-parser gpt-oss moves analysis into reasoning_content and
  # --tool-call-parser gpt-oss turns "commentary to=functions.X" into tool_calls.
  # Reasoning cannot be switched off (sglang rejects reasoning_effort "none"); clients
  # pick low / medium / high via reasoning_effort, default medium.
  #
  # Three local overrides (build the patched files with sglang-patches/apply_patches.sh):
  #  1. GPT_OSS_TEMPLATE: the model's own chat template reads past reasoning from
  #     message.thinking, but sglang keeps it in reasoning_content, so the analysis a
  #     client echoed back never reached the model (same prompt_tokens with or without
  #     it). The copy reads reasoning_content too, and renders content + reasoning on a
  #     tool-call turn (analysis + commentary preamble) where the original raises.
  #     It also drops the original's 50-character cut-off that rendered every array of
  #     objects in a tool schema as any[], hiding the item fields from the model.
  #     Diff: diff $MODEL_PATH/$MODEL_NAME/chat_template.jinja $GPT_OSS_TEMPLATE
  #  2. SGLANG_PATCH_DIR, two files bound over the image copies (pinned to the 0.5.20
  #     image; each has its .orig next to it for a diff):
  #     gpt_oss_detector.py -- sglang's grammar constraint for harmony ends generation
  #     right after the analysis message, so tool_choice required/named and tools with
  #     strict=true returned an empty reply with no tool call. The patch skips the
  #     constraint and parses the model's own tool call, as "auto" does. The forced
  #     call is therefore not grammar-enforced.
  #     harmony_parser.py -- the stock parser accepts a tool call only as
  #     "<|channel|>commentary to=functions.X <|constrain|>json". The model sometimes
  #     writes it on the analysis channel, with "code" or no content type, or with the
  #     recipient before the channel (9 of 16000 requests on the stock parser); those
  #     calls came back as an empty reply or as raw harmony text in content. The patch
  #     rewrites every header that has a functions.* recipient into the stock shape.
  #     Rarer shapes found by sampling 116000 raw outputs are handled in the same two
  #     files: no <|message|> token between header and arguments, recipient written as
  #     functions.X.json, arguments in parentheses, and the call written as a JSON body
  #     {"name": X, "arguments": {...}}.
  #     A message that ends in <|call|> and names no function cannot become a tool
  #     call (about 1 in 30000 replies). Its text is returned as content (commentary
  #     channel) or as reasoning (analysis channel), with no harmony tokens in it.
  #  3. HARMONY_CACHE: /v1/responses loads the o200k_harmony vocab over HTTPS on init
  #     and compute nodes may have no route out, so sglang disabled that API. The dir
  #     holds the vocab file pre-fetched on a login node.
  REQUIRED_NUM_GPUS=2
  MODEL_TAG="openai/gpt-oss-120b"
  MODEL_NAME="openai/gpt-oss-120b"   # dir under $MODEL_PATH (bound to /models)
  CTX_SIZE=131072     # model's max_position_embeddings
  SGLANG_PORT=$BACKEND_PORT   # internal; litellm front door forwards here
  SGLANG_SIF_PATH=$IMAGE_DIR/sglang_v0.5.20.sif
  GPT_OSS_TEMPLATE=$PATCH_DIR/gpt-oss/chat_template.jinja
  SGLANG_PATCH_DIR=$PATCH_DIR/sglang-v0.5.20
  # HARMONY_CACHE comes from the environment (env.example)
}

serve_sglang() {
  SGLANG_APPTAINER_OPTS=(
    --bind "$GPT_OSS_TEMPLATE:/gpt-oss-chat_template.jinja:ro"
    --bind "$SGLANG_PATCH_DIR/gpt_oss_detector.py:/sgl-workspace/sglang/python/sglang/srt/function_call/gpt_oss_detector.py:ro"
    --bind "$SGLANG_PATCH_DIR/harmony_parser.py:/sgl-workspace/sglang/python/sglang/srt/parser/harmony_parser.py:ro"
    --bind "$HARMONY_CACHE:/harmony-cache:ro"
    --env TIKTOKEN_RS_CACHE_DIR=/harmony-cache
  )
  run_sglang_single \
    --model-path "/models/$MODEL_NAME" \
    --served-model-name "$MODEL_TAG" \
    --chat-template /gpt-oss-chat_template.jinja \
    --reasoning-parser gpt-oss \
    --tool-call-parser gpt-oss \
    --tp "$REQUIRED_NUM_GPUS" \
    --enable-cache-report \
    --enable-metrics \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$SGLANG_PORT" \
    --context-length "$CTX_SIZE"
}
