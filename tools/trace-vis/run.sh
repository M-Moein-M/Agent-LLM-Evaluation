#!/usr/bin/env bash
# Start the trace viewer.
#
#   ./run.sh                       # workspace = the repo this script lives in
#   ./run.sh -p 9000               # pick a port
#   ./run.sh -w /other/workspace   # point at another checkout
#   ./run.sh --open                # bind 0.0.0.0 instead of localhost
#
# The workspace is bound READ-ONLY: the viewer never writes to traces/.
set -euo pipefail

HERE="$(dirname "$(readlink -f "$0")")"
SIF="$HERE/trace-vis.sif"
WORKSPACE="$(readlink -f "$HERE/../..")"
PORT=""
HOST="127.0.0.1"
EXTRA=()

usage() {
    sed -n '2,10p' "$0" | sed 's/^# \?//'
    echo
    echo "options: -p PORT  -w WORKSPACE  -s SIF  --open  --scan-fs"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)      PORT="$2"; shift 2 ;;
        -w|--workspace) WORKSPACE="$(readlink -f "$2")"; shift 2 ;;
        -s|--sif)       SIF="$(readlink -f "$2")"; shift 2 ;;
        --open)         HOST="0.0.0.0"; shift ;;
        -h|--help)      usage 0 ;;
        *)              EXTRA+=("$1"); shift ;;
    esac
done

[[ -f "$SIF" ]] || { echo "error: $SIF not found — run ./build.sh first" >&2; exit 1; }
[[ -d "$WORKSPACE/traces" ]] || {
    echo "error: $WORKSPACE has no traces/ — pass -w /path/to/repo" >&2; exit 1; }

# first free port from 8080 up, unless one was given
if [[ -z "$PORT" ]]; then
    for p in $(seq 8080 8130); do
        if ! (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then PORT=$p; break; fi
        exec 3<&- 2>/dev/null || true
    done
    [[ -n "$PORT" ]] || { echo "error: no free port in 8080-8130" >&2; exit 1; }
fi

echo "workspace : $WORKSPACE (read-only)"
echo "image     : $SIF"
echo "url       : http://localhost:$PORT"
if [[ "$HOST" == "127.0.0.1" ]]; then
    echo "remote?   : ssh -N -L $PORT:localhost:$PORT $(whoami)@$(hostname -f 2>/dev/null || hostname)"
else
    echo "note      : bound on 0.0.0.0 — reachable from other hosts on this network"
fi
echo

exec apptainer run \
    --containall \
    --bind "$WORKSPACE:/workspace:ro" \
    "$SIF" \
    --workspace /workspace --host "$HOST" --port "$PORT" "${EXTRA[@]+"${EXTRA[@]}"}"
