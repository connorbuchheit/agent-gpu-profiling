#!/usr/bin/env bash
# Start vLLM server (OpenAI-compat) on port 8000.
# If you see "Free memory ... is less than desired": another process (e.g. old vLLM) is using the GPU.
#   Kill it first:  pkill -f vllm.entrypoints.openai.api_server
#   Then run this script again.

set -e
export VLLM_USE_MODELSCOPE=false

# Qwen2.5-3B: 32k context. Use --gpu-memory-utilization 0.5 if GPU is shared.
python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2.5-3B-Instruct \
  --gpu-memory-utilization 0.85 \
  "$@"
