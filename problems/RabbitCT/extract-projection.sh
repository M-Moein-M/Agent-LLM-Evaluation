#!/bin/bash
# Extract a single projection image from RabbitInput.rct as a PNG.
# Usage: ./extract-projection.sh <image-index>

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <image-index>" >&2
  exit 1
fi

K=$1
RCT=${RCT:-RabbitInput/RabbitInput.rct}

if [ ! -f "$RCT" ]; then
  echo "Input file not found: $RCT" >&2
  exit 1
fi

read W H N < <(od -An -tu4 -N12 "$RCT")

if [ "$K" -lt 0 ] || [ "$K" -ge "$N" ]; then
  echo "Index $K out of range [0, $((N - 1))]" >&2
  exit 1
fi

IMG=$((W * H * 4))
OFF=$((20 + (K + 1) * 96 + K * IMG))

RAW=$(mktemp -t rct-proj.XXXXXX)
trap 'rm -f "$RAW"' EXIT

dd if="$RCT" of="$RAW" bs=1 skip="$OFF" count="$IMG" status=none

OUT="proj_${K}.png"
magick -size "${W}x${H}" -depth 32 -define quantum:format=floating-point \
       -endian LSB "gray:$RAW" -auto-level "$OUT"

echo "Wrote $OUT (projection $K of $N, ${W}x${H})"
