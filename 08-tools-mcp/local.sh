#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
cache="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}/models--Qwen--Qwen3-0.6B/snapshots"
for model in "$cache"/*; do
  if [ -f "$model/model.safetensors" ]; then
    exec uv run --offline --no-sync --with 'mcp==1.26.0' python 08-tools-mcp/demo.py --model "$model" "$@"
  fi
done
echo "找不到已缓存的 Qwen3-0.6B 权重：$cache" >&2
exit 1
