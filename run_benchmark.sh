#!/usr/bin/env bash
# Run inference matrix benchmark from repo root (requires vLLM/SGLang + deps).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  exec "${ROOT}/.venv/bin/python" "${ROOT}/benchmark.py" "$@"
fi
exec python3 "${ROOT}/benchmark.py" "$@"
