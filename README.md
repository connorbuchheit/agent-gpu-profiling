# agent-gpu-profiling

Prototype for profiling **GPU and memory usage** of a minimal LLM agent across **vLLM** and **SGLang** under different task profiles (short tool loops vs. longer multi-turn rollouts).

**See [PLANNING.md](PLANNING.md)** for goals, architecture, task taxonomy, and implementation phases.  
**Running on a Slurm cluster?** See [docs/SLURM.md](docs/SLURM.md).

## Quick start

1. **Install deps** (from repo root):
   ```bash
   pip install -r requirements.txt
   ```

2. **Start an LLM server** (vLLM or SGLang) on port 8000, e.g.:
   ```bash
   chmod +x scripts/run_vllm.sh && ./scripts/run_vllm.sh
   # or: ./scripts/run_sglang.sh
   ```

3. **Profile with live GPU/memory per step** (CLI):
   ```bash
   PYTHONPATH=src python -m agent_gpu_profiling.cli profile
   ```
   Optional: `--task-type short_loop|long_multiturn|mixed`, `--config path/to/config.yaml`.

4. **Run the full benchmark** (writes CSV to disk):
   ```bash
   python scripts/run_benchmark.py
   ```
   Metrics are written to `./results/` (configurable in `config/default.yaml`).

## Config

- `config/default.yaml`: backend, `base_url`, model, task types, sampling interval, output dir.
- Env overrides: `AGENT_GPU_BASE_URL`, `AGENT_GPU_MODEL`, `AGENT_GPU_OUTPUT_DIR`.
- **OpenAI API**: Put `OPENAI_API_KEY` in `.env` (gitignored). Set `base_url` to `https://api.openai.com/v1` and `model` to e.g. `gpt-4o-mini` to use it.

## Layout

- `src/agent_gpu_profiling/`: config, tasks (types + scenarios), agent runner, GPU profiler, harness, CLI.
- `scripts/`: `run_benchmark.py`, `run_vllm.sh`, `run_sglang.sh`.
- `config/default.yaml`: default config.