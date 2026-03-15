"""CLI: profile GPU and memory utilization across agent steps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure package is importable when run as __main__
if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_gpu_profiling.agent.runner import run_task
from agent_gpu_profiling.config import load_config
from agent_gpu_profiling.profiler.gpu import sample_gpu_once
from agent_gpu_profiling.tasks import TaskType, get_scenario


def _print_step_metrics(step: int, total: int) -> None:
    sample = sample_gpu_once()
    if sample is None:
        print(f"  Step {step}/{total}   GPU: N/A   Mem: N/A (pynvml not available or no GPU)")
        return
    util = sample.get("gpu_util_pct")
    used = sample.get("gpu_mem_used_mb")
    total_mb = sample.get("gpu_mem_total_mb")
    util_str = f"{util}%" if util is not None else "N/A"
    mem_str = f"{used:.0f} / {total_mb:.0f} MiB" if used is not None and total_mb is not None else "N/A"
    print(f"  Step {step}/{total}   GPU: {util_str}   Mem: {mem_str}")


def cmd_profile(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    base_url = config["base_url"]
    model = config["model"]
    backend = config.get("backend", "vllm")
    task_type_name = args.task_type or config.get("task_types", ["short_loop"])[0]
    short_steps = config.get("short_loop_steps", 5)
    long_turns = config.get("long_multiturn_turns", 3)

    try:
        task_type = TaskType(task_type_name)
    except ValueError:
        print(f"Unknown task type: {task_type_name}", file=sys.stderr)
        return 1

    scenario = get_scenario(
        task_type,
        num_steps=short_steps,
        num_turns=long_turns,
    )
    total_steps = len([t for t in scenario if t.get("role") == "user"])

    print(f"Profiling: backend={backend}  task_type={task_type_name}  steps={total_steps}")
    print("-" * 50)

    def on_step(step: int, total: int) -> None:
        _print_step_metrics(step, total)

    run_task(
        scenario,
        base_url=base_url,
        model=model,
        task_type=task_type,
        on_step=on_step,
    )

    print("-" * 50)
    print("Done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile GPU and memory utilization across agent steps.",
        prog="agent-gpu-profile",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser("profile", help="Run agent and print GPU/memory per step")
    profile_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config/default.yaml)",
    )
    profile_parser.add_argument(
        "--task-type",
        choices=[t.value for t in TaskType],
        default=None,
        help="Task type to run (default: first from config)",
    )
    profile_parser.set_defaults(func=cmd_profile)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
