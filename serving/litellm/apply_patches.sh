#!/bin/bash
# Apply patches/streaming_iterator-1.89.4.patch to the LiteLLM installed in a venv.
# usage: apply_patches.sh [VENV_DIR]        (default: $LITELLM_VENV, then $SERVE_ROOT/.venv-litellm)
#        SITE_PACKAGES=/path apply_patches.sh   patch a plain site-packages tree instead
#        apply_patches.sh --revert [VENV_DIR]   undo the patch
# Idempotent: an already patched tree is reported and left alone. A pip reinstall of
# litellm restores the unpatched file.
#
# What the patch fixes (two changes to the Anthropic /v1/messages streaming adapter):
#  1. When a streamed reply switches block type (text -> thinking, thinking -> text) upstream
#     emits content_block_stop + content_block_start and then discards the chunk that
#     triggered the switch unless it holds tool arguments. For backends that stream
#     reasoning_content (sglang) every thinking and text block lost its first chunk.
#  2. sglang sends a "\n" content chunk between two tool calls, sometimes before the last
#     argument chunk of the first call. Upstream closed the tool_use block for it, so the
#     late chunk opened a nameless tool_use block and the first call lost its closing brace.
set -euo pipefail

revert=0
if [ "${1:-}" = --revert ]; then revert=1; shift; fi
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
patch_file=$here/patches/streaming_iterator-1.89.4.patch
want=1.89.4

if [ -n "${SITE_PACKAGES:-}" ]; then
  sp=$SITE_PACKAGES
else
  venv=${1:-${LITELLM_VENV:-${SERVE_ROOT:-$HOME/llm-serving}/.venv-litellm}}
  [ -x "$venv/bin/python" ] || { echo "no python in $venv, pass the LiteLLM venv dir" >&2; exit 1; }
  sp=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
fi

ver=$(ls -d "$sp"/litellm-*.dist-info 2>/dev/null | sed 's|.*/litellm-||; s|\.dist-info||' | head -n1)
if [ "$ver" != "$want" ]; then
  echo "found litellm '${ver:-none}' in $sp, the patch targets $want only" >&2
  exit 1
fi

if [ "$revert" = 1 ]; then
  patch -p1 -R -d "$sp" <"$patch_file"
  echo "reverted"
elif patch -p1 -R --dry-run -s -d "$sp" <"$patch_file" >/dev/null 2>&1; then
  echo "already patched: $sp"
else
  patch -p1 --dry-run -d "$sp" <"$patch_file"
  patch -p1 -d "$sp" <"$patch_file"
  echo "patched: $sp"
fi
