#! /bin/bash
# Generate code for the problem

PROBLEM_DIR=$1

mkdir -p /root/.pi/agent
cd "$PROBLEM_DIR"
rm -rf .git/

tree -a
tree /root/.pi -a

# Non-interactive mode (process and exit)
#      pi -p "List all .ts files in src/"
pi --list-models
# PROMPT="write a OBSERVATION.md file that briefly explains this repo"
PROMPT="The simulation is not working properly when using simd. Please debug and fix the issue. Just analyze and fix the code. Don't execute the code."
pi -p  --mode json --model Qwen/Qwen3.6-35B-A3B-FP8 $PROMPT | jq -c 'select(.type == "agent_end")' | tee ~/agent.jsonl
cp ~/agent.jsonl "$PROBLEM_DIR"
