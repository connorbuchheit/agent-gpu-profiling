#!/usr/bin/env bash
# Example: start vLLM server (OpenAI-compat) on port 8000.
# Install: pip install vllm
# Adjust model and GPU args as needed.

set -e
export VLLM_USE_MODELSCOPE=false

# Qwen2.5-3B: 32k context; or Qwen/Qwen2.5-7B-Instruct for heavier (needs more VRAM)
python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2.5-3B-Instruct \
  "$@"
