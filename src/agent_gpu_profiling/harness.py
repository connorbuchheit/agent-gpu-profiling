"""Orchestrate: load config, run each task type, collect GPU metrics per run."""

from __future__ import annotations

import uuid
from pathlib import Path

from agent_gpu_profiling.agent.runner import run_task
from agent_gpu_profiling.agent.tools import TOOLS
from agent_gpu_profiling.config import load_config
from agent_gpu_profiling.profiler.gpu import GPUMetricsCollector
from agent_gpu_profiling.tasks import TaskType, get_scenario


def run_harness(config_path: str | Path | None = None) -> list[dict]:
    """
    Load config, run each configured task type once, collect GPU metrics during each run.
    Returns a list of run summaries (backend, task_type, run_id, output_csv path).
    """
    config = load_config(config_path)
    base_url = config["base_url"]
    model = config["model"]
    backend = config.get("backend", "vllm")
    task_types = config.get("task_types", ["short_loop", "long_multiturn"])
    interval = config.get("sampling_interval_sec", 1.0)
    out_dir = Path(config["output_dir"])
    short_steps = config.get("short_loop_steps", 5)
    long_turns = config.get("long_multiturn_turns", 3)
    reasoning_turns = config.get("long_reasoning_turns", 25)
    shared_prefix_steps = config.get("shared_prefix_steps", 12)

    summaries = []
    for tt_name in task_types:
        try:
            task_type = TaskType(tt_name)
        except ValueError:
            continue
        scenario = get_scenario(
            task_type,
            num_steps=short_steps,
            num_turns=long_turns,
            num_reasoning_turns=reasoning_turns,
            num_shared_prefix_steps=shared_prefix_steps,
        )
        run_id = str(uuid.uuid4())[:8]
        csv_path = out_dir / f"metrics_{backend}_{tt_name}_{run_id}.csv"
        collector = GPUMetricsCollector(
            csv_path,
            interval_sec=interval,
            backend=backend,
            task_type=tt_name,
            run_id=run_id,
        )
        collector.start()
        try:
            run_task(
                scenario,
                base_url=base_url,
                model=model,
                task_type=task_type,
                tools=TOOLS if task_type == TaskType.TOOL_LOOP else None,
            )
        finally:
            collector.stop()
        summaries.append({
            "backend": backend,
            "task_type": tt_name,
            "run_id": run_id,
            "metrics_csv": str(csv_path),
        })
    return summaries
