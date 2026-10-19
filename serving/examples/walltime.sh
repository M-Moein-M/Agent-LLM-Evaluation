#!/bin/bash
# Prints the sbatch --time/--time-min arguments for a job submitted now: 24h normally, but
# ending MARGIN_MIN before the next announced maintenance so Slurm does not hold the job
# ("ReqNodeNotAvail, Reserved for maintenance"). Exits 1 if less than 4h is left.
# usage: MAINTENANCE_AT="2026-09-30 11:00" walltime.sh      (no variable = plain 24h)
# Copy the two values into TIME_LIMIT / TIME_MIN in serving/env.
DOWNTIME=${MAINTENANCE_AT:-}
MARGIN_MIN=${MARGIN_MIN:-15}
MAX_MIN=$((24 * 60))
MIN_MIN=$((4 * 60))

if [ -z "$DOWNTIME" ]; then
  printf -- "--time=24:00:00\n"
  exit 0
fi
left=$(( ($(date -d "$DOWNTIME" +%s) - $(date +%s)) / 60 - MARGIN_MIN ))
if [ "$(date +%s)" -ge "$(date -d "$DOWNTIME" +%s)" ] || [ "$left" -ge "$MAX_MIN" ]; then
  mins=$MAX_MIN
elif [ "$left" -ge "$MIN_MIN" ]; then
  mins=$left
else
  echo "walltime.sh: only ${left}min before maintenance $DOWNTIME (< ${MIN_MIN}min)" >&2
  exit 1
fi
printf -- "--time=%d:%02d:00" $((mins / 60)) $((mins % 60))
# Near a maintenance, --time-min lets Slurm shrink further if the job starts later than now.
[ "$mins" -lt "$MAX_MIN" ] && printf -- " --time-min=%d:00:00" $((MIN_MIN / 60))
echo
