#!/bin/bash
# stop_stack.sh RUNDIR : kill the PIDs recorded by start_stack.sh, nothing else.
while read -r name pid; do kill "$pid" 2>/dev/null && echo "killed $name $pid"; done <"$1/pids"
sleep 2
while read -r _ pid; do
  if kill -0 "$pid" 2>/dev/null; then echo "still alive $pid, SIGKILL"; kill -9 "$pid"; fi
done <"$1/pids"
