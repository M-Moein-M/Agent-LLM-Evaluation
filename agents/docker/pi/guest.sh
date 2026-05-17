#! /bin/bash
# Generate code for the problem

PROBLEM_DIR=$1

mkdir -p /root/.pi/agent
cd "$PROBLEM_DIR"

tree -a
tree /root/.pi -a

# Non-interactive mode (process and exit)
#      pi -p "List all .ts files in src/"
pi --list-models
pi -p  --mode json --model Qwen/Qwen3.6-35B-A3B-FP8 "write a OBSERVATION.md file that briefly explains this repo" | jq -c 'select(.type == "agent_end")' | tee ~/agent.log
cp ~/agent.log "$PROBLEM_DIR"
