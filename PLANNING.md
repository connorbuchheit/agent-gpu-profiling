# Agent GPU Profiling — Planning & Reference

Prototype for profiling **GPU and memory usage** of a basic LLM agent across different **server backends** (vLLM, SGLang) under different **task profiles** (short tool loops vs. longer multi-turn rollouts).

---

## 1. Goals

| Goal | Description |
|------|-------------|
| **Compare backends** | Run the same agent workload against vLLM and SGLang and collect comparable metrics. |
| **Task-aware profiling** | Define task types that stress different query patterns (many short calls vs. fewer long conversations). |
| **Metrics** | GPU utilization, GPU memory (allocated/used), and optionally CPU/RAM, over time and per task. |
| **Reproducibility** | Config-driven: model, backend, task suite, and output paths so runs are repeatable. |

---

## 2. Infrastructure Choices (LLM Servers)

| Backend | Role | Notes |
|---------|------|--------|
| **vLLM** | High-throughput, PagedAttention | Strong for batch/throughput; simple to deploy (`pip install vllm`). |
| **SGLang** | RadixAttention, multi-turn/agentic | Often better for structured output and multi-turn; good prefix caching. |

Both expose **OpenAI-compatible** (or similar) HTTP APIs, so the agent client can stay backend-agnostic by using a single API interface (e.g. `openai` client with `base_url`).

**Implementation note:** For a prototype, we assume the LLM server is **already running** (launched separately). This repo focuses on the **agent client**, **task definitions**, and **profiling harness** that drive and measure usage.

---

## 3. Task Taxonomy (Query Rollouts)

Different “rollouts” exercise how each backend handles:

- **Short, frequent requests** (tool loops): many small prompts, quick back-and-forth.
- **Longer, fewer requests** (multi-turn): fewer calls, longer context and responses.

Suggested task types:

| Task type | Description | Typical pattern |
|-----------|-------------|-----------------|
| **Short tool loop** | Agent does N small steps (e.g. “call tool A → call tool B → answer”). | Many short requests; lower tokens per request. |
| **Long multi-turn** | Simulated conversation or long chain-of-thought with fewer, bigger turns. | Fewer requests; higher tokens per request and larger context. |
| **Mixed** | Combination of short and long segments in one run. | Stress both patterns in one session. |

These become **scenarios** we replay against each backend while collecting GPU/memory samples.

---

## 4. Metrics to Collect

- **GPU**
  - Utilization (%).
  - Memory: allocated (reserved), used (actually in use).
- **Optional**
  - CPU utilization, system RAM (for context).
  - Per-request: latency (TTFT, time to last token), token counts.

**How:** Use existing tools and keep the prototype simple:

- **nvidia-smi** (or `pynvml`) for GPU utilization and memory.
- Sample at a fixed interval (e.g. every 1–2 s) during a task run; tag samples by task id and backend.
- Persist to CSV/JSON for later comparison and plotting.

---

## 5. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Profiling harness (this repo)                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ Task suite  │  │ Agent runner │  │ Metrics collector        │ │
│  │ (scenarios) │→ │ (tool loops  │→ │ (GPU/mem sampling +      │ │
│  │             │  │ or multi-turn)│  │  per-run logs)          │ │
│  └─────────────┘  └──────┬───────┘  └────────────┬────────────┘ │
│                          │                        │              │
│                          ▼                        ▼              │
│                   OpenAI-compat API          nvidia-smi / pynvml │
└──────────────────────────┬────────────────────────┬─────────────┘
                           │                        │
┌──────────────────────────▼────────────────────────▼─────────────┐
│  External: vLLM or SGLang server (one at a time) + GPU(s)       │
└─────────────────────────────────────────────────────────────────┘
```

- **Config** selects: backend URL, model name, which task types to run, sampling interval, output dir.
- **Agent** is minimal: call LLM (completion or chat), optionally parse “tool” calls and re-call; no heavy framework required for the prototype.
- **Profiler** starts sampling when a task starts and stops when it ends; aggregates per (backend, task_type, run_id).

---

## 6. Implementation Phases

### Phase 1 — Foundation (current)
- [ ] Project layout and config (e.g. `config.yaml` or env + small Python config).
- [ ] Task type definitions (data structures or small classes for “short_loop”, “long_multiturn”, “mixed”).
- [ ] Stub agent runner: single function that takes a scenario and calls the LLM server (OpenAI-compat); optional simple tool loop (fixed number of steps).
- [ ] Profiler module: spawn background thread or process to sample GPU (and optionally CPU) at interval; write raw samples to CSV/JSON per run.

### Phase 2 — Backends and scenarios
- [ ] Document how to run vLLM and SGLang locally (or in Docker) with the same model.
- [ ] Implement 2–3 concrete scenarios (e.g. “5-step tool loop”, “3-turn long conversation”, “mixed”).
- [ ] Run harness against one backend; verify metrics and logs.

### Phase 3 — Comparison and reporting
- [ ] Run full suite against vLLM and SGLang (same model, same machine).
- [ ] Add a small script or notebook to load saved metrics and produce comparison tables/plots (GPU util, memory over time, per-task summaries).

### Phase 4 — Optional extensions
- [ ] Add more task types or real tool implementations.
- [ ] Support multiple models or batch sizes.
- [ ] Integrate with existing benchmark tools (e.g. SGLang’s bench serving, or custom load scripts).

---

## 7. Project Layout (target)

```
agent-gpu-profiling/
├── PLANNING.md           # This file
├── README.md
├── requirements.txt      # or pyproject.toml
├── config/
│   └── default.yaml      # backend URL, model, task list, sampling interval
├── src/
│   └── agent_gpu_profiling/
│       ├── __init__.py
│       ├── config.py     # load config
│       ├── tasks/
│       │   ├── __init__.py
│       │   ├── types.py  # TaskType enum / dataclasses
│       │   └── scenarios.py  # concrete scenario definitions
│       ├── agent/
│       │   ├── __init__.py
│       │   └── runner.py # minimal agent (LLM + optional tool loop)
│       ├── profiler/
│       │   ├── __init__.py
│       │   └── gpu.py    # sample GPU (and optionally CPU), save to file
│       └── harness.py    # orchestrate: load config → run tasks → collect metrics
├── scripts/
│   ├── run_vllm.sh       # example: start vLLM server
│   ├── run_sglang.sh     # example: start SGLang server
│   └── run_benchmark.py  # entrypoint: run harness with config
└── results/              # output dir for metrics (gitignored or committed as samples)
```

---

## 8. Config (example)

- `backend`: `vllm` | `sglang` (or just `base_url` and model).
- `model`: e.g. `meta-llama/Llama-3.2-1B` (or whatever the server exposes).
- `task_types`: list of task type ids to run (e.g. `["short_loop", "long_multiturn"]`).
- `sampling_interval_sec`: e.g. `1.0`.
- `output_dir`: e.g. `./results`.

---

## 9. References

- vLLM: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) — OpenAI-compatible server.
- SGLang: [sgl-project.github.io](https://sgl-project.github.io/) — RadixAttention, good for multi-turn/agent; OpenAI-compat.
- GPU metrics: `nvidia-smi`, Python `pynvml` for programmatic sampling.

Use this doc as the single source of truth when implementing each phase; update the checklists as you go.
