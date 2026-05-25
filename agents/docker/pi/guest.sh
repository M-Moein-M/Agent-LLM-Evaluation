#! /bin/bash
# Generate code for the problem

PROBLEM_DIR=$1

mkdir -p /root/.pi/agent
cd "$PROBLEM_DIR"
rm -rf .git/
rm -rf .github/

# store the prompt for agent and remove it from the problem directory
PROMPT=$(cat PROMPT.md)
rm PROMPT.md

tree -a
tree /root/.pi -a

# Non-interactive mode (process and exit)
#      pi -p "List all .ts files in src/"
pi --list-models
pi -p  --mode json --model Qwen/Qwen3.6-35B-A3B-FP8 "$PROMPT" | jq -c 'select(.type == "agent_end")' | tee ~/agent.jsonl
cp ~/agent.jsonl "$PROBLEM_DIR"
