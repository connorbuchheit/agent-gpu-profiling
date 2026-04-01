# SGLang support and RadixAttention experiments

This repo talks to **any OpenAI-compatible** HTTP API. SGLang’s server exposes `/v1/chat/completions` the same way vLLM does, so the **same client** works for both.

## Run SGLang (aligned with vLLM)

1. Install: `pip install "sglang[all]"` (or your environment’s equivalent).

2. Start the server (defaults match `scripts/run_vllm.sh` model/port for apples-to-apples runs):

   ```bash
   bash scripts/run_sglang.sh
   ```

   Override model or port without editing the script:

   ```bash
   SGLANG_MODEL=Qwen/Qwen2.5-3B-Instruct SGLANG_PORT=8000 bash scripts/run_sglang.sh
   ```

3. Point config at SGLang:

   ```yaml
   backend: sglang
   base_url: "http://localhost:8000/v1"
   model: "Qwen/Qwen2.5-3B-Instruct"   # must match --model-path
   ```

   Or via env:

   ```bash
   export AGENT_GPU_BASE_URL=http://localhost:8000/v1
   export AGENT_GPU_MODEL=Qwen/Qwen2.5-3B-Instruct
   export AGENT_GPU_BACKEND=sglang
   ```

4. Run the profiler:

   ```bash
   PYTHONPATH=src python3 -m agent_gpu_profiling.cli profile --task-type shared_prefix
   ```

## Robustness checklist

| Item | Notes |
|------|--------|
| **Model string** | Must match what SGLang serves (`curl -s http://localhost:8000/v1/models`). |
| **base_url** | Must include `/v1` (same as OpenAI client convention). |
| **Port** | If SGLang uses another port, set `base_url` / `AGENT_GPU_BASE_URL` accordingly. |
| **Tool calling** | `tool_loop` needs a model + SGLang build that supports OpenAI-style tools. |

## Investigating RadixAttention (prefix cache)

**RadixAttention** reuses KV for shared prefixes (multi-turn, repeated system/doc, many branches). Effects show up as:

- **Lower latency** on later steps when the prompt’s prefix is largely unchanged and cache hits.
- **SGLang server logs** often report prefix / radix cache hit rate (exact field depends on version—watch stderr while profiling).

### Built-in workload: `shared_prefix`

Task type **`shared_prefix`** (`TaskType.SHARED_PREFIX`):

- Merges a **long fixed system document** (synthetic “reference doc” repeated for length) with the default system prompt.
- Then runs **`shared_prefix_steps`** short user questions (default 12, set in `config/default.yaml`).

The chat history **grows** each turn, but the **initial long block** is shared structure Radix-style caches are good at. Compare:

1. **SGLang** — run `shared_prefix`, note per-step **latency** and **prompt token** growth in the CLI summary / CSV.
2. **vLLM** — same config and model, same task; PagedAttention behaves differently for prefix reuse.
3. **Optional A/B on SGLang** — start server once with default (radix on), once with cache disabled **if** your `sglang.launch_server --help` lists a disable flag; then compare CSVs.

### What to look for

- **Latency vs step index**: later steps may get faster on SGLang if prefix hits dominate prefill cost.
- **Prompt tokens**: should increase roughly linearly with turns; completion tokens may stay small.
- **Server logs**: correlate “prefix cache hit rate” (or similar) with latency dips.

This repo does **not** reimplement RadixAttention; it provides a **repeatable agent-shaped workload** and **aligned metrics** (GPU, memory, latency, tokens) so you can quantify impact when switching backends or server flags.
