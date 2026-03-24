"""CLI: profile GPU and memory utilization across agent steps, with optional graph and CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Ensure package is importable when run as __main__
if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_gpu_profiling.agent.runner import run_task
from agent_gpu_profiling.agent.tools import TOOLS
from agent_gpu_profiling.config import load_config
from agent_gpu_profiling.profiler.gpu import sample_gpu_once
from agent_gpu_profiling.tasks import TaskType, get_scenario

try:
    import plotext as plt
except ImportError:
    plt = None  # type: ignore


def _write_csv(samples: list[dict], path: Path) -> None:
    """Write step, gpu_util_pct, gpu_mem_used_mb, latency_ms, prompt_tokens, completion_tokens to CSV."""
    if not samples:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "gpu_util_pct", "gpu_mem_used_mb", "latency_ms", "prompt_tokens", "completion_tokens"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(samples)


def _plot_terminal(samples: list[dict], task_type_name: str) -> None:
    """Draw a simple terminal graph of GPU % and memory over steps (requires plotext)."""
    if not plt or not samples:
        return
    steps = [s["step"] for s in samples]
    util = [s.get("gpu_util_pct") or 0 for s in samples]
    mem = [s.get("gpu_mem_used_mb") or 0 for s in samples]
    plt.clf()
    plt.subplots(2, 1)
    plt.subplot(1, 1)
    plt.plot(steps, util, label="GPU %")
    plt.title(f"GPU utilization by step ({task_type_name})")
    plt.xlabel("Step")
    plt.ylabel("GPU %")
    plt.theme("clear")
    plt.show()
    plt.subplot(2, 1)
    plt.plot(steps, mem, label="Mem (MiB)")
    plt.title(f"GPU memory used by step ({task_type_name})")
    plt.xlabel("Step")
    plt.ylabel("Mem MiB")
    plt.theme("clear")
    plt.show()


def cmd_profile(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    base_url = config["base_url"]
    model = config["model"]
    backend = config.get("backend", "vllm")
    task_type_name = args.task_type or config.get("task_types", ["short_loop"])[0]
    short_steps = config.get("short_loop_steps", 5)
    long_turns = config.get("long_multiturn_turns", 3)
    reasoning_turns = getattr(args, "reasoning_turns", None) or config.get("long_reasoning_turns", 25)
    shared_prefix_steps = config.get("shared_prefix_steps", 12)
    out_dir = Path(config.get("output_dir", "./results"))
    show_graph = not getattr(args, "no_graph", False)

    try:
        task_type = TaskType(task_type_name)
    except ValueError:
        print(f"Unknown task type: {task_type_name}", file=sys.stderr)
        return 1

    scenario = get_scenario(
        task_type,
        num_steps=short_steps,
        num_turns=long_turns,
        num_reasoning_turns=reasoning_turns,
        num_shared_prefix_steps=shared_prefix_steps,
    )
    use_tools = task_type == TaskType.TOOL_LOOP
    total_steps = len([t for t in scenario if t.get("role") == "user"]) if not use_tools else "N (tool rounds)"

    samples: list[dict] = []

    def on_step(step: int, total: int, step_data: dict | None = None) -> None:
        sample = sample_gpu_once()
        if sample is None:
            sample = {}
        util = sample.get("gpu_util_pct")
        used = sample.get("gpu_mem_used_mb")
        total_mb = sample.get("gpu_mem_total_mb")
        util_str = f"{util}%" if util is not None else "N/A"
        mem_str = f"{used:.0f} / {total_mb:.0f} MiB" if used is not None and total_mb is not None else "N/A"
        lat_ms = (step_data or {}).get("latency_ms")
        pt = (step_data or {}).get("prompt_tokens")
        ct = (step_data or {}).get("completion_tokens")
        lat_str = f"{lat_ms}ms" if lat_ms is not None else "N/A"
        tok_str = f"{pt}+{ct}" if pt is not None and ct is not None else (f"{pt}+?" if pt is not None else "N/A")
        print(f"  Step {step}/{total}   GPU: {util_str}   Mem: {mem_str}   Latency: {lat_str}   Tokens: {tok_str}")
        row = {"step": step, "gpu_util_pct": util, "gpu_mem_used_mb": used}
        if lat_ms is not None:
            row["latency_ms"] = lat_ms
        if pt is not None:
            row["prompt_tokens"] = pt
        if ct is not None:
            row["completion_tokens"] = ct
        samples.append(row)

    print(f"Profiling: backend={backend}  task_type={task_type_name}  steps={total_steps}")
    print("-" * 50)

    run_task(
        scenario,
        base_url=base_url,
        model=model,
        task_type=task_type,
        on_step=on_step,
        tools=TOOLS if use_tools else None,
    )

    print("-" * 50)

    csv_path = out_dir / f"profile_{backend}_{task_type_name}.csv"
    _write_csv(samples, csv_path)
    print(f"Wrote: {csv_path}")

    # Summary stats
    if samples:
        utils = [s.get("gpu_util_pct") for s in samples if s.get("gpu_util_pct") is not None]
        mems = [s.get("gpu_mem_used_mb") for s in samples if s.get("gpu_mem_used_mb") is not None]
        lats = [s.get("latency_ms") for s in samples if s.get("latency_ms") is not None]
        pt_sum = sum(s.get("prompt_tokens") or 0 for s in samples)
        ct_sum = sum(s.get("completion_tokens") or 0 for s in samples)
        n = len(samples)
        print("\n--- Summary ---")
        if utils:
            print(f"  GPU %     min/avg/max: {min(utils):.0f} / {sum(utils)/n:.0f} / {max(utils):.0f}")
        if mems:
            print(f"  Mem MiB   min/avg/max: {min(mems):.0f} / {sum(mems)/n:.0f} / {max(mems):.0f}")
        if lats:
            print(f"  Latency   min/avg/max: {min(lats)} / {round(sum(lats)/n)} / {max(lats)} ms")
        if pt_sum or ct_sum:
            print(f"  Tokens   prompt: {pt_sum}   completion: {ct_sum}   total: {pt_sum + ct_sum}")

    if show_graph and samples and plt:
        print("\n--- GPU utilization over steps ---")
        _plot_terminal(samples, task_type_name)
    elif show_graph and samples and not plt:
        print("(Install plotext for terminal graph: pip install plotext)")

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
    profile_parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip terminal graph (still write CSV)",
    )
    profile_parser.add_argument(
        "--reasoning-turns",
        type=int,
        default=None,
        metavar="N",
        help="Override long_reasoning_turns (for --task-type long_reasoning). Default from config (e.g. 25).",
    )
    profile_parser.set_defaults(func=cmd_profile)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
