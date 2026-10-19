#!/bin/bash
# start_stack.sh RUNDIR MOCK_PORT PROXY_PORT
# Start the mock backend and the LiteLLM proxy (both on 127.0.0.1) without any GPU or
# sglang: the proxy uses the same rendered config and callback as the real stack, its
# backend is tests/mock_backend.py. PIDs go to RUNDIR/pids (stop with stop_stack.sh).
# Then run:  python3 tests/proxy_probe.py http://127.0.0.1:PROXY_PORT RUNDIR LABEL
#
# env: LITELLM_VENV (default $SERVE_ROOT/.venv-litellm, or set LITELLM_BIN),
#      LITELLM_MASTER_KEY (optional; export PROXY_API_KEY to the same value for the probe),
#      LITELLM_PROMETHEUS=1 to include the prometheus callback.
set -euo pipefail
RUN=$(mkdir -p "$1" && cd "$1" && pwd)
MP=$2
PP=$3
SERVING_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SERVING_DIR
. "$SERVING_DIR/common/lib.sh"
. "$SERVING_DIR/common/litellm.sh"
load_env
VENV_PY=$(dirname "$LITELLM_BIN")/python

MODEL_TAG=${PROBE_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}
render_litellm_config >"$RUN/config.yaml"
cp "$SERVING_DIR/litellm/reasoning_bridge.py" "$RUN/"
cd "$RUN"
: >"$RUN/pids"

"$VENV_PY" -I "$SERVING_DIR/tests/mock_backend.py" "$MP" "$RUN/backend.jsonl" >"$RUN/mock.log" 2>&1 &
echo "mock $!" >>"$RUN/pids"

# Same environment start_litellm exports for a real job.
export LITELLM_MODEL_NAME="$MODEL_TAG" LITELLM_BACKEND_MODEL="openai/$MODEL_TAG" \
  LITELLM_API_BASE="http://127.0.0.1:$MP/v1" LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES=1 \
  LITELLM_DROP_PARAMS=1 LITELLM_MODEL_NAME_THINK="$MODEL_TAG-think" LITELLM_CTX_SIZE=256000 PYTHONUTF8=1
# LiteLLM 1.104.x refuses to start without a master key: local tests only.
if [ -z "${LITELLM_MASTER_KEY:-}" ] && [ -n "${ALLOW_NO_MASTER_KEY:-}" ]; then
  export LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY=true
fi
"$LITELLM_BIN" --config "$RUN/config.yaml" --host 127.0.0.1 --port "$PP" >"$RUN/proxy.log" 2>&1 &
PROXY_PID=$!
echo "proxy $PROXY_PID" >>"$RUN/pids"

n=0
until curl -sf -m 2 "http://127.0.0.1:$PP/health/liveliness" >/dev/null 2>&1 || [ "$n" -ge 45 ] || ! kill -0 "$PROXY_PID" 2>/dev/null; do
  sleep 2
  n=$((n + 1))
done
echo "proxy up after $((n * 2))s: $(curl -s -m 2 "http://127.0.0.1:$PP/health/liveliness")"
