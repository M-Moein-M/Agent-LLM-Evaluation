#!/bin/bash
# Build the patched sglang files (and the patched gpt-oss chat template) that the model
# files bind over the image copies.
#
# usage: apply_patches.sh <v0.5.16|v0.5.20|gpt-oss> [SIF_OR_MODEL_FILE]
#   v0.5.NN   pristine files are read out of $IMAGE_DIR/sglang_v0.5.NN.sif (or the given
#             .sif), the patches in sglang-patches/v0.5.NN are applied to a temp copy, and
#             <name>.py plus <name>.py.orig are written to $PATCH_DIR/sglang-v0.5.NN/.
#   gpt-oss   pristine template from $MODEL_DIR/openai/gpt-oss-120b/chat_template.jinja (or
#             the given file) -> $PATCH_DIR/gpt-oss/chat_template.jinja (+ .orig).
# Patches are plain unified diffs with a/ b/ paths relative to /sgl-workspace/sglang (or to
# the template). A patch that does not apply cleanly means the image is not the version the
# patch targets: see the README next to the patch.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
serving=$(dirname "$here")
# shellcheck disable=SC1091
. "$serving/common/lib.sh"
load_env

what=${1:?usage: apply_patches.sh <v0.5.16|v0.5.20|gpt-oss> [SIF_OR_MODEL_FILE]}
override=${2:-}
img_root=/sgl-workspace/sglang   # sglang checkout inside the images

tree=$(mktemp -d)
trap 'rm -rf "$tree"' EXIT

case $what in
  v*)
    sif=${override:-$IMAGE_DIR/sglang_$what.sif}
    out=$PATCH_DIR/sglang-$what
    [ -f "$sif" ] || { echo "image not found: $sif" >&2; exit 1; }
    ;;
  gpt-oss)
    src=${override:-$MODEL_DIR/openai/gpt-oss-120b/chat_template.jinja}
    out=$PATCH_DIR/gpt-oss
    [ -f "$src" ] || { echo "template not found: $src" >&2; exit 1; }
    ;;
  *) echo "unknown target '$what'" >&2; exit 1 ;;
esac
mkdir -p "$out"

for p in "$here/$what"/*.patch; do
  rel=$(sed -n 's|^+++ b/||p' "$p" | head -n1)
  mkdir -p "$tree/$(dirname "$rel")"
  if [ "$what" = gpt-oss ]; then
    cp "$src" "$tree/$rel"
  else
    apptainer exec "$sif" cat "$img_root/$rel" >"$tree/$rel"
  fi
  cp "$tree/$rel" "$out/$(basename "$rel").orig"
  patch -p1 -d "$tree" <"$p"
  cp "$tree/$rel" "$out/$(basename "$rel")"
  echo "wrote $out/$(basename "$rel") (+ .orig)"
done
