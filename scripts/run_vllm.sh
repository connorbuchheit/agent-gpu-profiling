#!/usr/bin/env bash
# Example: start vLLM server (OpenAI-compat) on port 8000.
# Install: pip install vllm
# Adjust model and GPU args as needed.

set -e
export VLLM_USE_MODELSCOPE=false

# Use a small model for local prototyping, e.g.:
#   --model meta-llama/Llama-3.2-1B
# Or TinyLlama:
#   --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model meta-llama/Llama-3.2-1B \
  "$@"
