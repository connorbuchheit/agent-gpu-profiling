# Running agent-gpu-profiling on the Slurm cluster

You're **user12**. Don't run GPU workloads on login nodes — use `srun` or `sbatch` to get a compute node.

## 1. Connect and get your repo on the cluster

```bash
# From your laptop (use one of the login nodes)
ssh user12@35.84.33.219
# or: user12@44.230.162.249, etc.
```

Clone or upload this repo into your home (e.g. `~/agent-gpu-profiling`). Home is on shared storage, so it’s visible from any node.

## 2. One-time setup (on a compute node or login for pip only)

You need Python deps and vLLM (or SGLang) on the cluster. Easiest: get an interactive GPU node and do setup there:

```bash
# Get one GPU interactively (run from a login node)
srun --gpus=1 --mem=64G --pty bash
# You’re now on a compute node

cd ~/agent-gpu-profiling
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install vllm
# Optional: pip install "sglang[all]" if you want SGLang too
```

If you use the **container** instead:

```bash
srun --gpus=1 --mem=64G --container-image=nvcr.io#nvidia/pytorch:24.03-py3 --pty bash
cd ~/agent-gpu-profiling   # if home is mounted
source .venv/bin/activate   # if you already created venv on shared storage
# or: pip install -r requirements.txt && pip install vllm
```

(Your venv lives in home, so once created it’s available from any session that has access to home.)

**If `python3 -m venv` fails** (e.g. “ensurepip is not available” and you don’t have sudo): skip the venv and use the system Python with user installs. On the compute node:

```bash
cd ~/agent-gpu-profiling
pip install --user -r requirements.txt
pip install --user vllm
# Optional: export PATH="$HOME/.local/bin:$PATH"
```

Then run the server and profiler with `python3` (see section 3 below; don’t `source .venv/bin/activate`).

**Alternative: use the cluster container** (often has a working Python/venv):

```bash
srun --gpus=1 --mem=64G --container-image=nvcr.io#nvidia/pytorch:24.03-py3 --pty bash
cd ~/agent-gpu-profiling
pip install -r requirements.txt vllm
# Then run server + profiler with python (no venv needed if container has pip)
```

## 3. Run the agent + profiler on GPU

You need the **model server** (vLLM) and the **client** (this repo) on the **same** node. Two ways:

### Option A: Interactive (good for testing)

From a **login node**, start an interactive GPU session:

```bash
srun --gpus=1 --mem=64G --pty bash
```

On that **compute node**:

```bash
cd ~/agent-gpu-profiling
# If you have a venv: source .venv/bin/activate
# If you used pip --user: no activate needed, use python3

# Start vLLM in the background (uses the GPU)
./scripts/run_vllm.sh &
sleep 30   # give it time to load the model

# Run the profiler (hits localhost:8000, so inference is on this GPU)
export PYTHONPATH=src
python3 -m agent_gpu_profiling.cli profile
# or: python3 scripts/run_benchmark.py

# When done, stop vLLM
kill %1
exit
```

### Option B: Batch job (hands-off)

From a **login node**, from the repo directory:

```bash
cd ~/agent-gpu-profiling
mkdir -p logs
sbatch scripts/slurm_profile.sh
```

This requests 1 GPU, starts vLLM, runs the benchmark, and writes logs under `logs/`. See `scripts/slurm_profile.sh` for resource and time limits.

## 4. Quick reference

| Where        | Do this |
|-------------|--------|
| Login node  | SSH, edit code, `sbatch` / `srun` (no GPU work). |
| Compute node| Run vLLM/SGLang + this repo’s client (GPU workload). |
| Interactive | `srun --gpus=1 --mem=64G --pty bash` then run server + client. |
| Batch       | `sbatch scripts/slurm_profile.sh` from repo root. |

Cluster details you had: NVIDIA Driver 590.48.01, CUDA 13.1, 32 GPU nodes, 61 TB Lustre home.
