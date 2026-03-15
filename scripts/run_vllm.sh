#!/usr/bin/env bash
# Example: start vLLM server (OpenAI-compat) on port 8000.
# Install: pip install vllm
# Adjust model and GPU args as needed.

set -e
export VLLM_USE_MODELSCOPE=false

# TinyLlama = no Hugging Face login; or use meta-llama/Llama-3.2-1B with HF_TOKEN
python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  "$@"
