#!/bin/bash
# Publish the serving node's endpoint URL to a file the agent jobs poll.
#
# Run this ON THE SERVING NODE, after (or alongside) the LiteLLM/sglang start,
# from inside the serving job:
#
#   bash /path/to/nhr-eval-workspace/scripts/publish_endpoint.sh dsflash0731 30000
#
# Why: the agent jobs sit in the a40 queue for hours and cannot know which node
# the serving job will land on. Hardcoding an IP means every serving restart
# needs the agent job cancelled and resubmitted (sbatch spools the script at
# submit time, so editing the file on disk does NOT retarget a queued job).
# Instead the serving side writes its address here and the agent side re-reads
# it on every gate iteration.
#
# The file lives in the workspace on /home, which is shared across the
# clusters -- unlike $WORK, which only exists on cluster.
#
# Usage: publish_endpoint.sh <name> [port] [ip]
#   name  logical endpoint name, e.g. dsflash0731  -> endpoints/<name>.url
#   port  default 30000
#   ip    default: this node's address on the 10.0.0.1/16 fabric

set -uo pipefail

NAME="${1:?usage: publish_endpoint.sh <name> [port] [ip]}"
PORT="${2:-30000}"
IP="${3:-}"

WS="${ENDPOINT_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DIR="$WS/endpoints"
FILE="$DIR/$NAME.url"

if [ -z "$IP" ]; then
    # Prefer the cluster fabric address; the first hostname -I entry is often a
    # management or loopback-ish interface the other cluster cannot reach.
    IP=$(ip -4 -o addr show scope global 2>/dev/null \
         | awk '{print $4}' | cut -d/ -f1 | grep '^10\.28\.' | head -1)
    [ -z "$IP" ] && IP=$(hostname -I 2>/dev/null | awk '{print $1}')
fi
[ -z "$IP" ] && { echo "ERROR: could not determine an IP; pass one explicitly" >&2; exit 2; }

URL="http://$IP:$PORT/v1"

mkdir -p "$DIR"
# Write-then-rename so a reader never sees a half-written line.
TMP="$FILE.tmp.$$"
{
    echo "$URL"
    echo "# host=$(hostname -s) job=${SLURM_JOB_ID:-none} written=$(date -Is)"
} > "$TMP"
mv -f "$TMP" "$FILE"

echo "published $URL -> $FILE"
