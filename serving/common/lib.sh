#!/bin/bash
# Shared helpers for launch.sh and both serve.sbatch files. Source it, do not run it.
#
# Every command with a side effect (apptainer, srun, rsync, mkdir, ...) goes through
# run / run_bg / run_bg_log. With DRY_RUN=1 these print the fully expanded command and
# do nothing, otherwise they execute it. The commands are built the same way in both
# cases, so the dry run shows exactly what a real run would execute.

SERVING_DIR=${SERVING_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export SERVING_DIR

log() { printf '[serve] %s\n' "$*"; }
die() { printf '[serve] ERROR: %s\n' "$*" >&2; exit 1; }
is_dry() { [ "${DRY_RUN:-0}" = 1 ]; }

# Shell precedence: environment > $SERVING_ENV (default serving/env) > serving/env.example.
# All three files assign with ${VAR:-default}, so the first writer wins.
load_env() {
  set -a
  # shellcheck disable=SC1090
  [ -f "${SERVING_ENV:-$SERVING_DIR/env}" ] && . "${SERVING_ENV:-$SERVING_DIR/env}"
  # shellcheck disable=SC1091
  . "$SERVING_DIR/env.example"
  set +a
  return 0
}

# ---------------------------------------------------------------------------
# command printing / execution
# ---------------------------------------------------------------------------
_q="'"
shq() {
  if [[ $1 =~ ^[A-Za-z0-9_@%+=:,./-]+$ ]]; then
    printf '%s' "$1"
  else
    printf "'%s'" "${1//$_q/$_q\\$_q$_q}"
  fi
}

# fmt_cmd cmd args...: line breaks before options, images and sub-commands, valid shell.
fmt_cmd() {
  local a first=1
  for a in "$@"; do
    if [ "$first" = 1 ]; then
      first=0
      shq "$a"
    elif [[ $a == --* || $a == *.sif || $a == apptainer || $a == sglang ]]; then
      printf ' \\\n    %s' "$(shq "$a")"
    else
      printf ' %s' "$(shq "$a")"
    fi
  done
}

# show_cmd SUFFIX cmd args...   (DRY_RUN_FORMAT=argv prints one argument per line, used by the parity test)
show_cmd() {
  local suffix=$1 a
  shift
  if [ "${DRY_RUN_FORMAT:-shell}" = argv ]; then
    printf 'CMD%s\n' "$suffix"
    for a in "$@"; do printf 'ARG\t%s\n' "$a"; done
  else
    printf '+ '
    fmt_cmd "$@"
    printf '%s\n' "$suffix"
  fi
}

run() {
  if is_dry; then show_cmd "" "$@"; return 0; fi
  "$@"
}

# Background variants set LAST_BG_PID (0 in a dry run).
run_bg() {
  if is_dry; then show_cmd " &" "$@"; LAST_BG_PID=0; return 0; fi
  "$@" &
  LAST_BG_PID=$!
}

# run_bg_log LOGFILE cmd...: background, stdout and stderr appended to LOGFILE.
run_bg_log() {
  local logfile=$1
  shift
  if is_dry; then show_cmd " >>$(shq "$logfile") 2>&1 &" "$@"; LAST_BG_PID=0; return 0; fi
  "$@" >>"$logfile" 2>&1 &
  LAST_BG_PID=$!
}

# export_env NAME VALUE: export in this shell, show it in a dry run.
export_env() {
  export "$1=$2"
  if is_dry; then
    if [ "${DRY_RUN_FORMAT:-shell}" = argv ]; then printf 'ENV\t%s=%s\n' "$1" "$2"; else printf '+ export %s=%s\n' "$1" "$(shq "$2")"; fi
  fi
  return 0
}

# ---------------------------------------------------------------------------
# slurm helpers
# ---------------------------------------------------------------------------

# Fake allocation of N nodes for DRY_RUN outside a job.
dry_slurm_env() {
  local n=${1:-1} i names=""
  for ((i = 1; i <= n; i++)); do names+="${names:+,}$(printf 'node%02d' "$i")"; done
  export SLURM_JOB_ID=${SLURM_JOB_ID:-0}
  export SLURM_JOB_NODELIST=$names
  export SLURM_JOB_NUM_NODES=$n
  export SLURM_NNODES=$n
  export SLURM_NODEID=0
}

list_nodes() {
  if is_dry || ! command -v scontrol >/dev/null 2>&1; then
    tr ',' '\n' <<<"$SLURM_JOB_NODELIST"
  else
    scontrol show hostnames "$SLURM_JOB_NODELIST"
  fi
}

this_host() {
  if is_dry; then list_nodes | head -n1; else hostname -s; fi
}

# Full host name of the node this task runs on (placeholder in a dry run).
node_hostname() {
  if is_dry; then list_nodes | head -n1; else hostname; fi
}

# nodes_for_gpus N: nodes needed for N GPUs at GPUS_PER_NODE.
nodes_for_gpus() { echo $((($1 + GPUS_PER_NODE - 1) / GPUS_PER_NODE)); }

# print_vars NAME...: resolved variables, one per line.
print_vars() {
  local v
  for v in "$@"; do printf '  %-22s = %s\n' "$v" "${!v-<unset>}"; done
}

# ---------------------------------------------------------------------------
# template rendering (prometheus.yml.tmpl)
#   lines starting with '#'        template comments, dropped
#   # @if VAR ... # @endif          block kept when VAR is non-empty (no nesting)
#   # @targets VAR                  one "- 'word'" list item per word of VAR
#   @{VAR}                          replaced by the value of VAR
# ---------------------------------------------------------------------------
render_template() {
  local line name w skip=0
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in
      '# @if '*)
        name=${line#'# @if '}
        if [ -z "${!name:-}" ]; then skip=1; fi
        continue
        ;;
      '# @endif'*) skip=0; continue ;;
      '# @targets '*)
        name=${line#'# @targets '}
        if [ "$skip" = 0 ]; then
          # shellcheck disable=SC2086
          for w in ${!name}; do printf "          - '%s'\n" "$w"; done
        fi
        continue
        ;;
      '#'*) continue ;;
    esac
    if [ "$skip" = 1 ]; then continue; fi
    while [[ $line =~ @\{([A-Za-z_][A-Za-z0-9_]*)\} ]]; do
      name=${BASH_REMATCH[1]}
      line=${line//"@{$name}"/${!name:-}}
    done
    printf '%s\n' "$line"
  done <"$1"
}
