#!/bin/bash
set -euo pipefail

# Download RabbitInput.tgz (~2.9 GB) and extract it into the target data dir.
# Non-interactive (no confirmation prompt) so it can be driven by run_eval.sh.
#
# Usage: download-input.sh [DEST_DIR]
#   DEST_DIR defaults to the repo-root data/RabbitCT so the script also works
#   standalone. The archive expands to <DEST_DIR>/RabbitInput/.

URL="https://hpc-mover.rrze.uni-erlangen.de/HPC-Data/0x7b58aefb/eig7ahyo6fo2bais0ephuf2aitohv1ai/RabbitInput.tgz"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$SCRIPT_DIR/../../data/RabbitCT}"

mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"   # absolutise

echo "Downloading RabbitInput.tgz (~2.9 GB) into $DEST ..."
curl -L -o "$DEST/RabbitInput.tgz" "$URL"

echo "Extracting..."
tar -xzf "$DEST/RabbitInput.tgz" -C "$DEST"

echo "Removing archive..."
rm "$DEST/RabbitInput.tgz"

echo "Done. Data in $DEST/RabbitInput"
