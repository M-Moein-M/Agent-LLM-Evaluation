#!/bin/bash
# verify_run.sh HOST MODEL_TAG [WORKERS]
# Prefix-cache + metrics + Prometheus battery against a RUNNING stack on HOST (the node that
# runs the batch script). WORKERS = number of router workers, 0 (default) for a plain
# `sglang serve`. Output goes to stdout and to $OUT (default ./verify-results.txt).
#
# Checks: /v1/models on the backend port, sglang_* series on every engine /metrics port,
# sgl_router_* on the router metrics port (fleet only), Prometheus targets and the number of
# distinct sglang_* series, then cached_tokens in the usage block for three calls with the same
# ~700-token prefix straight to the backend, two non-stream calls through LiteLLM and one
# stream call through LiteLLM. cached_tokens > 0 from call 2 on means the prefix cache works.
#
# env: BACKEND_PORT (30060) LITELLM_PORT (30000) PROM_PORT (9090) API_KEY (dummy; the
#      LITELLM_MASTER_KEY if you set one; raw sglang ignores it)
set -u
HOST=${1:?usage: verify_run.sh HOST MODEL_TAG [WORKERS]}
TAG=${2:?usage: verify_run.sh HOST MODEL_TAG [WORKERS]}
NW=${3:-0}
BACKEND_PORT=${BACKEND_PORT:-30060}
LITELLM_PORT=${LITELLM_PORT:-30000}
PROM_PORT=${PROM_PORT:-9090}
API_KEY=${API_KEY:-dummy}
OUT=${OUT:-verify-results.txt}
: >"$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }

# ~700-token shared prefix (repeated filler), distinct short user turn per call.
PREFIX=$(python3 -c "print(('The quick brown fox jumps over the lazy dog. ' * 90).strip())")

chat_req() { # $1=base_url  $2=user_msg   (auth header is harmless for raw sglang)
  curl -sm 60 "$1/v1/chat/completions" \
    -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' -d @- <<JSON
{"model":"$TAG","max_tokens":8,"temperature":0,
 "messages":[{"role":"system","content":"$PREFIX"},{"role":"user","content":"$2"}]}
JSON
}
usage_line() {
  python3 -c "import sys,json
try:
 d=json.load(sys.stdin); u=d.get('usage') or {}
 det=u.get('prompt_tokens_details') or {}
 print('prompt_tokens=%s cached_tokens=%s'%(u.get('prompt_tokens'),det.get('cached_tokens')))
except Exception as e:
 print('parse-error:',e)"
}

log "================ $TAG on $HOST ($NW router worker(s)) ================"

log "--- /v1/models (backend :$BACKEND_PORT) ---"
curl -sm10 "http://$HOST:$BACKEND_PORT/v1/models" | jq -c '.data[]?|{id,root}' 2>/dev/null | tee -a "$OUT" || log "  NO RESPONSE"

log "--- engine sglang_ series on worker /metrics ---"
if [ "$NW" -gt 0 ]; then ports=$(seq $((BACKEND_PORT + 1)) $((BACKEND_PORT + NW))); else ports=$BACKEND_PORT; fi
for p in $ports; do
  n=$(curl -sm10 "http://$HOST:$p/metrics" | grep -c '^sglang' || true)
  log "  :$p  -> $n sglang_* samples"
done
if [ "$NW" -gt 0 ]; then
  n=$(curl -sm10 "http://$HOST:$((BACKEND_PORT + 100))/metrics" | grep -c '^sgl_router' || true)
  log "  :$((BACKEND_PORT + 100)) (router) -> $n sgl_router_* samples"
fi

log "--- Prometheus :$PROM_PORT targets ---"
curl -sm10 "http://$HOST:$PROM_PORT/api/v1/targets" |
  jq -r '.data.activeTargets[]?|"  \(.labels.job)  \(.scrapeUrl)  health=\(.health)  \(.lastError//"")"' |
  tee -a "$OUT" || log "  NO RESPONSE"
log "--- Prometheus: # of distinct sglang_* metric names ---"
curl -sm10 "http://$HOST:$PROM_PORT/api/v1/query" --data-urlencode 'query=count by (__name__)({__name__=~"sglang_.+"})' |
  jq -r '.data.result|length as $l|"  \($l) distinct sglang_* metrics"' | tee -a "$OUT"
curl -sm10 "http://$HOST:$PROM_PORT/api/v1/query" --data-urlencode 'query=sum(sglang_num_running_reqs) or sum(sglang:num_running_reqs)' |
  jq -c '.data.result' | sed 's/^/  num_running_reqs: /' | tee -a "$OUT"

log "--- cached_tokens: 3x same 700-tok prefix, direct :$BACKEND_PORT ---"
for i in 1 2 3; do
  r=$(chat_req "http://$HOST:$BACKEND_PORT" "ping number $i")
  log "  call $i: $(printf '%s' "$r" | usage_line)"
  sleep 3
done
log "--- cached_tokens: 2x via LiteLLM :$LITELLM_PORT (non-stream) ---"
for i in 1 2; do
  r=$(chat_req "http://$HOST:$LITELLM_PORT" "litellm ping $i")
  log "  call $i: $(printf '%s' "$r" | usage_line)"
  sleep 3
done
log "--- cached_tokens: stream via LiteLLM :$LITELLM_PORT (include_usage) ---"
curl -sm60 "http://$HOST:$LITELLM_PORT/v1/chat/completions" -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$TAG\",\"max_tokens\":8,\"temperature\":0,\"stream\":true,\"stream_options\":{\"include_usage\":true},\"messages\":[{\"role\":\"system\",\"content\":\"$PREFIX\"},{\"role\":\"user\",\"content\":\"stream ping\"}]}" |
  grep -o '"prompt_tokens_details":[^}]*}' | tail -1 | sed 's/^/  stream usage: /' | tee -a "$OUT"
log "done"
