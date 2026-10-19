#!/bin/bash
# LiteLLM front door (default port 30000) for all clients. Needs lib.sh sourced first.
# LiteLLM owns the public port and sglang sits behind it on an internal port. Every client
# connects to :30000 and never changes when the model changes. LiteLLM runs on the host
# (python venv), sglang in the container, both on the batch node's localhost.

# render_litellm_config: template -> stdout, resolving the "@" markers.
#   @callbacks    adds "prometheus" when LITELLM_PROMETHEUS=1
#   @master_key   lines are kept when LITELLM_MASTER_KEY is set, removed otherwise
render_litellm_config() {
  local cb='"reasoning_bridge.proxy_handler_instance"'
  if [ "${LITELLM_PROMETHEUS:-0}" = 1 ]; then cb+=', "prometheus"'; fi
  local tmpl=$SERVING_DIR/litellm/config.template.yaml
  if [ -n "${LITELLM_MASTER_KEY:-}" ]; then
    sed -e "s|^\\(  callbacks: \\).*# @callbacks\$|\\1[$cb]|" -e 's|  # @master_key$||' "$tmpl"
  else
    sed -e "s|^\\(  callbacks: \\).*# @callbacks\$|\\1[$cb]|" -e '/# @master_key$/d' "$tmpl"
  fi
}

# start_litellm BACKEND_PORT: runs in the foreground (callers background it).
# Reads MODEL_TAG and optionally CTX_SIZE.
start_litellm() {
  local backend_port=$1
  local cfg_dir=$RUN_DIR/$SLURM_JOB_ID/litellm cfg
  cfg=$cfg_dir/config.yaml
  log "Starting LiteLLM front door on :$LITELLM_PORT -> sglang :$backend_port ..."
  # The litellm config is model-agnostic: it reads these env vars via os.environ/...
  # so switching models is just selecting a different models/<name>.sh (no yaml edit).
  export_env LITELLM_MODEL_NAME "$MODEL_TAG"            # public id on /v1/models
  export_env LITELLM_BACKEND_MODEL "openai/$MODEL_TAG"  # forwarded as served-model-name
  export_env LITELLM_API_BASE "http://localhost:$backend_port/v1"
  export_env LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES 1
  export_env LITELLM_DROP_PARAMS 1
  export_env LITELLM_MODEL_NAME_THINK "${MODEL_TAG}-think"
  # Reported as max_input_tokens on /model_group/info. Unset for models without CTX_SIZE.
  if [ -n "${CTX_SIZE:-}" ]; then export_env LITELLM_CTX_SIZE "$CTX_SIZE"; fi

  run mkdir -p "$cfg_dir"
  if is_dry; then
    log "[litellm] config $cfg rendered from litellm/config.template.yaml (callbacks, master_key markers resolved)"
    if [ "${DRY_RUN_VERBOSE:-0}" = 1 ]; then render_litellm_config | sed 's/^/    | /'; fi
  else
    render_litellm_config >"$cfg"
    cp "$SERVING_DIR/litellm/reasoning_bridge.py" "$cfg_dir/"
  fi

  # PYTHONUTF8=1: litellm prints a unicode startup banner; with the job's latin-1
  # default stdout (redirected to the log) that banner would crash startup.
  LITELLM_CMD=(env PYTHONUTF8=1 "$LITELLM_BIN"
    --config "$cfg"
    --host 0.0.0.0
    --port "$LITELLM_PORT")
  run "${LITELLM_CMD[@]}"
}
