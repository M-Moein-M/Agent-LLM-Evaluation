#!/bin/bash
# usage: health.sh <jobid> <model-tag>
# Waits until the Slurm job runs, then until the stack answers a real completion through
# LiteLLM (its /health is OK before sglang is up, so a completion is required; the "*"
# catch-all in the config routes any model name to the backend). Prints the endpoint.
# If ENDPOINT_FILE is set, the line "<model-tag>:<ip>" in that file is updated in place.
# env: LITELLM_PORT (30000), NODE_DOMAIN_SUFFIX (appended to the node name for the lookup,
#      for example .example.org), API_KEY (dummy)
JOB=${1:?usage: health.sh <jobid> <model-tag>}
TAG=${2:?usage: health.sh <jobid> <model-tag>}
PORT=${LITELLM_PORT:-30000}
API_KEY=${API_KEY:-dummy}
while :; do
  L=$(squeue -h -j "$JOB" -o "%T|%N" 2>/dev/null)
  [ -z "$L" ] && { echo "job $JOB gone before healthy"; exit 1; }
  [ "${L%%|*}" = RUNNING ] && break
  sleep 30
done
NODE=${L#*|}
NODE=${NODE%%,*}   # batch node = first node
IP=$(getent ahostsv4 "$NODE${NODE_DOMAIN_SUFFIX:-}" | awk 'NR==1{print $1}')
echo "RUNNING on $NODE / $IP"
until curl -sf -m 120 "http://$IP:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -H "Authorization: Bearer $API_KEY" \
    -d "{\"model\":\"$TAG\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with only: OK\"}]}" >/dev/null; do
  squeue -h -j "$JOB" | grep -q . || { echo "job $JOB died while waiting for health"; exit 1; }
  sleep 30
done
if [ -n "${ENDPOINT_FILE:-}" ]; then
  python3 - "$ENDPOINT_FILE" "$TAG" "$IP" <<'PY'
import sys
f, t, ip = sys.argv[1:]
lines = open(f).read().splitlines()
hit = False
for i, l in enumerate(lines):
    if l.split(':', 1)[0].strip() == t:
        lines[i] = f"{t}:{ip}"
        hit = True
open(f, 'w').write("\n".join(lines) + ("\n" if lines else ""))
print("wrote" if hit else "tag not present in file, not written")
PY
fi
echo "HEALTHY $TAG $NODE $IP http://$IP:$PORT"
