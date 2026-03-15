#!/usr/bin/env bash
# Example: start SGLang server (OpenAI-compat) on port 8000.
# Install: pip install "sglang[all]"
# Adjust model as needed.

set -e

# Use same model as vLLM for fair comparison
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model-path meta-llama/Llama-3.2-1B \
  "$@"
