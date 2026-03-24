#!/usr/bin/env bash
# Start SGLang OpenAI-compatible server (same port/model pattern as run_vllm.sh for fair comparison).
# Install: pip install "sglang[all]"
#
# API: http://0.0.0.0:8000/v1  (set base_url in config to http://localhost:8000/v1)
# Model id in requests must match --model-path (often the HF repo id).
#
# RadixAttention: enabled by default in recent SGLang. To compare cache on vs off, run twice:
#   1) This script as-is, then profile with --task-type shared_prefix
#   2) If your SGLang version supports it, add e.g. --disable-radix-cache (see: python3 -m sglang.launch_server --help)
#
# If port 8000 is taken, use another port and set AGENT_GPU_BASE_URL, e.g.:
#   export AGENT_GPU_BASE_URL=http://127.0.0.1:30000/v1

set -e

MODEL="${SGLANG_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${SGLANG_PORT:-8000}"

python3 -m sglang.launch_server \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --model-path "${MODEL}" \
  "$@"
