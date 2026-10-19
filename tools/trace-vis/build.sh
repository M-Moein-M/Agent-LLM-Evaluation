#!/usr/bin/env bash
# Build the trace-vis Apptainer image.
#
#   ./build.sh              -> trace-vis.sif
#   ./build.sh /tmp/tv.sif  -> that path instead
#
# Needs network only to pull the python:3.12-slim base layer; the frontend build
# and the selftest gate inside %post are offline.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

SIF="${1:-trace-vis.sif}"
DEF="trace-vis.def"

command -v apptainer >/dev/null || { echo "error: apptainer not on PATH" >&2; exit 1; }

# keep the layer cache off $HOME — it is small here and fills up
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${TMPDIR:-/tmp}/apptainer-cache}"
mkdir -p "$APPTAINER_CACHEDIR"

# fail fast on the host before spending a build on it
echo "== host-side selftest =="
python3 selftest.py

echo "== apptainer build -> $SIF =="
if [[ -e "$SIF" ]]; then
    mv -f "$SIF" "$SIF.bak-$(date +%Y%m%d-%H%M%S)"
    echo "   (previous image moved aside)"
fi
apptainer build --fakeroot "$SIF" "$DEF"

echo
echo "built: $(readlink -f "$SIF")"
echo "start it with: ./run.sh"
