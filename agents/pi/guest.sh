#! /bin/bash
# Generate code for the problem

PROBLEM_DIR=$1
mkdir -p ~/.pi/agent
cp /models.json ~/.pi/agent/
cd $PROBLEM_DIR
tree -a

tree ~/.pi -a

# Non-interactive mode (process and exit)
#      pi -p "List all .ts files in src/"
pi --list-models
pi -p --model local/llama3.1:8b "finish the code in main.cpp" 2>&1 | tee ./agent.log