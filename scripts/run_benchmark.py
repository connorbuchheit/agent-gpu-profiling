#!/usr/bin/env python3
"""Entrypoint: run the profiling harness. Ensure LLM server is up and PYTHONPATH includes src."""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root / src so "agent_gpu_profiling" resolves
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from agent_gpu_profiling.harness import run_harness


def main():
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    summaries = run_harness(config_path)
    for s in summaries:
        print(f"  {s['backend']} / {s['task_type']} -> {s['metrics_csv']}")
    print(f"Done. {len(summaries)} run(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
