#!/bin/bash -l
# Re-run ONLY the measure phase against already-evacuated solve output.
#
# Every agents/apptainer/*/host.sh now rsyncs the solved code out to its
# permanent TRACE_DIR right after solve_job finishes (before measure runs),
# and wraps measure's apptainer exec in EVAL_MEASURE_TIMEOUT (default 1800s).
# A round that times out there already has complete source at TRACE_DIR, just
# missing .eval/facts.json (marked with a .measure_timeout file). This script
# replays just the measure step against that evacuated code -- intended to run
# standalone, e.g. on a fresh node/allocation after the campaign that produced
# it has ended.
#
# measure.sh is agent-agnostic (one canonical copy, see agents/measure.sh), so
# this works uniformly across claude/kimi/pi/swe-mini.
#
# Usage:
#   scripts/remeasure.sh <trace-dir> [<trace-dir> ...]
#   scripts/remeasure.sh --scan traces/     # auto-finds round-* dirs that are
#                                            # missing facts.json and were NOT
#                                            # a solve timeout (nothing to
#                                            # measure in that case)
#   scripts/remeasure.sh --perf-only <trace-dir> [...]
#                                            # only the paired perf bench of an
#                                            # already measured round: adds
#                                            # .eval/perf_paired.log and changes
#                                            # nothing else in the trace
#
# Env overrides: EVAL_MEASURE_TIMEOUT (1800), EVAL_SIZE (256),
#   EVAL_VARIANT (LolaBunnyGPU), MEASURE_LOCK, DATA_ROOT, EVAL_GPU (auto).
#
# Run from repo root.

set -uo pipefail

ROOT="$(pwd)"
[[ -d "$ROOT/agents" && -d "$ROOT/problems" ]] || { echo "ERROR: run from repo root" >&2; exit 1; }

EVAL_SIZE="${EVAL_SIZE:-256}"
EVAL_VARIANT="${EVAL_VARIANT:-LolaBunnyGPU}"
EVAL_MEASURE_TIMEOUT="${EVAL_MEASURE_TIMEOUT:-1800}"
MEASURE_LOCK="${MEASURE_LOCK:-${TMPDIR:-/tmp}/agent-eval-measure.lock}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
MEASURE_SCRIPT="$ROOT/agents/measure.sh"

NV_FLAG=()
if [[ "${EVAL_GPU:-auto}" == "1" ]] || { [[ "${EVAL_GPU:-auto}" == "auto" ]] &&
      command -v nvidia-smi >/dev/null 2>&1 &&
      nvidia-smi -L 2>/dev/null | grep -q "^GPU"; }; then
    NV_FLAG=(--nv)
fi
echo "GPU passthrough: ${NV_FLAG[*]:-none (CPU-only run)}"

PERF_ONLY=0
if [[ "${1:-}" == "--perf-only" ]]; then
    PERF_ONLY=1
    shift
fi

collect_dirs() {
    if [[ "${1:-}" == "--scan" ]]; then
        find "${2:-traces}" -mindepth 1 -type d -name 'round-*' 2>/dev/null | while IFS= read -r d; do
            [[ -f "$d/.solve_timeout" ]] && continue
            [[ -f "$d/.round_lost" ]] && continue
            [[ -s "$d/.eval/facts.json" ]] && continue
            echo "$d"
        done
    else
        printf '%s\n' "$@"
    fi
}

remeasure_one() {
    local TRACE_DIR="${1%/}"
    # traces/<problem>/<agent>/<model_tag>/<date>/round-<n>
    local ROUND_PART; ROUND_PART="$(basename "$TRACE_DIR")"
    local ROUND="${ROUND_PART#round-}"
    local DATE_DIR; DATE_DIR="$(dirname "$TRACE_DIR")"
    local MODEL_DIR; MODEL_DIR="$(dirname "$DATE_DIR")"
    local MODEL_TAG; MODEL_TAG="$(basename "$MODEL_DIR")"
    local AGENT_DIR; AGENT_DIR="$(dirname "$MODEL_DIR")"
    local AGENT; AGENT="$(basename "$AGENT_DIR")"
    local PROBLEM; PROBLEM="$(basename "$(dirname "$AGENT_DIR")")"

    local PROBLEM_SRC="$ROOT/problems/$PROBLEM"
    local AGENT_SIF="$ROOT/agents/apptainer/$AGENT/agent.sif"
    if [[ ! -f "$AGENT_SIF" ]]; then
        echo "ERROR: no agent.sif for agent '$AGENT' ($AGENT_SIF) -- bad trace-dir? $TRACE_DIR" >&2
        return 1
    fi
    if [[ ! -d "$PROBLEM_SRC" ]]; then
        echo "ERROR: unknown problem '$PROBLEM' -- bad trace-dir? $TRACE_DIR" >&2
        return 1
    fi

    # The trace holds only files the agent changed (host.sh rsyncs with
    # --compare-dest), so rebuild the work dir as problem source + trace.
    local WORK_DIR; WORK_DIR="$(mktemp -d)"
    rsync -a "$PROBLEM_SRC"/ "$WORK_DIR"/
    rsync -a "$TRACE_DIR"/ "$WORK_DIR"/
    if [[ "$PERF_ONLY" == "1" ]]; then
        rm -f "$WORK_DIR/.eval/perf_paired.log"
    else
        rm -rf "$WORK_DIR/.eval" "$WORK_DIR/.solve_timeout" "$WORK_DIR/.measure_timeout"
    fi

    local DATA_BIND=()
    if [[ -d "$DATA_ROOT/$PROBLEM/RabbitInput" ]]; then
        DATA_BIND=(--bind "$DATA_ROOT/$PROBLEM/RabbitInput:/problem/RabbitInput:ro")
    fi
    local GRADE_BIND=()
    if [[ -f "$ROOT/checks/$PROBLEM/bench.cu" ]]; then
        GRADE_BIND=(--bind "$ROOT/checks/$PROBLEM/bench.cu:/grade/bench.cu:ro")
    fi

    echo "[remeasure] $PROBLEM | $AGENT | $MODEL_TAG | round: $ROUND"
    (
        flock -x 200
        timeout "$EVAL_MEASURE_TIMEOUT" apptainer exec --cleanenv --pid "${NV_FLAG[@]}" \
            --env "EVAL_AGENT=$AGENT" \
            --env "EVAL_MODEL=$MODEL_TAG" \
            --env "EVAL_ROUND=$ROUND" \
            --env "EVAL_SIZE=$EVAL_SIZE" \
            --env "EVAL_VARIANT=$EVAL_VARIANT" \
            --env "EVAL_REQUIRE_GPU=${EVAL_REQUIRE_GPU:-0}" \
            --env "EVAL_PERF_ONLY=$PERF_ONLY" \
            --bind "$WORK_DIR:/problem" \
            --bind "$MEASURE_SCRIPT:/measure.sh:ro" \
            "${DATA_BIND[@]}" \
            "${GRADE_BIND[@]}" \
            "$AGENT_SIF" /bin/bash -l /measure.sh /problem
    ) 200>"$MEASURE_LOCK"
    local rc=$?

    if [[ "$PERF_ONLY" == "1" ]]; then
        local OUT="$WORK_DIR/.eval/perf_paired.log"
        local ok=1
        if [[ -s "$OUT" ]]; then
            cp "$OUT" "$TRACE_DIR/.eval/perf_paired.log"
            grep -q "^BENCH paired" "$OUT" && ok=0
            echo "[remeasure] perf-only rc=$rc: $TRACE_DIR"
            grep "^BENCH" "$OUT" | sed 's/^/    /'
        else
            echo "[remeasure] ERROR: perf-only rc=$rc, no perf_paired.log: $TRACE_DIR" >&2
        fi
        rm -rf "$WORK_DIR"
        return $ok
    fi

    if [ "$rc" -eq 124 ]; then
        echo "[remeasure] TIMEOUT again after ${EVAL_MEASURE_TIMEOUT}s: $TRACE_DIR" >&2
        touch "$TRACE_DIR/.measure_timeout"
        rm -rf "$WORK_DIR"
        return 124
    fi

    rsync -a --prune-empty-dirs --compare-dest="$PROBLEM_SRC/" "$WORK_DIR/" "$TRACE_DIR/"
    rm -rf "$WORK_DIR"

    if [ -s "$TRACE_DIR/.eval/facts.json" ]; then
        rm -f "$TRACE_DIR/.measure_timeout"
        echo "[remeasure] OK: $TRACE_DIR"
        return 0
    fi
    echo "[remeasure] ERROR: still no facts.json after re-run: $TRACE_DIR" >&2
    return 1
}

FAIL=0
while IFS= read -r d; do
    [[ -z "$d" ]] && continue
    remeasure_one "$d" || FAIL=1
done < <(collect_dirs "$@")

if [[ "$PERF_ONLY" != "1" ]] && command -v python3 >/dev/null 2>&1; then
    python3 "$ROOT/checks/check_pass.py" --traces "$ROOT/traces"
    python3 "$ROOT/checks/trace_stats.py" --traces "$ROOT/traces"
fi

exit $FAIL
