#!/bin/bash
#SBATCH --job-name=agent-gpu-profile
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Run from repo root:  sbatch scripts/slurm_profile.sh
# One-time: mkdir -p logs; pip install -r requirements.txt vllm (in a venv on the cluster)

set -e
mkdir -p logs

# Assume we're in repo root (e.g. cd ~/agent-gpu-profiling before sbatch)
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

# Activate venv if it exists (create it once with: python -m venv .venv && .venv/bin/pip install -r requirements.txt vllm)
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Start vLLM in background on this node's GPU
python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model meta-llama/Llama-3.2-1B &
VLLM_PID=$!

# Wait for server to be ready (model load can take 1–2 min)
echo "Waiting for vLLM to be ready..."
for i in $(seq 1 90); do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q 200; then
    echo "vLLM ready."
    break
  fi
  sleep 2
done

export PYTHONPATH=src
python scripts/run_benchmark.py

# Cleanup
kill $VLLM_PID 2>/dev/null || true
