#! /bin/bash
# Generate code for the problem

PROBLEM_DIR=$1

mkdir -p /root/.claude
cd "$PROBLEM_DIR"

claude -p "Preview this code repository and suggest improvements." --output-format json --verbose --allowedTools "Read,Edit,Bash" --permission-mode bypassPermissions > /output.json
