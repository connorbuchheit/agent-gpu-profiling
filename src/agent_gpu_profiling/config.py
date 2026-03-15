"""Load and validate config from YAML and env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config from YAML file. Uses config/default.yaml if path is None."""
    base = Path(__file__).resolve().parent.parent.parent
    load_dotenv(base / ".env")
    if path is None:
        path = base / "config" / "default.yaml"
    path = Path(path)
    if not path.exists():
        return _default_config()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    # Env overrides
    if base_url := os.environ.get("AGENT_GPU_BASE_URL"):
        data["base_url"] = base_url
    if model := os.environ.get("AGENT_GPU_MODEL"):
        data["model"] = model
    if out := os.environ.get("AGENT_GPU_OUTPUT_DIR"):
        data["output_dir"] = out
    return {**_default_config(), **data}


def _default_config() -> dict[str, Any]:
    return {
        "backend": "vllm",
        "base_url": "http://localhost:8000/v1",
        "model": "meta-llama/Llama-3.2-1B",
        "task_types": ["short_loop", "long_multiturn"],
        "sampling_interval_sec": 1.0,
        "output_dir": "./results",
        "short_loop_steps": 5,
        "long_multiturn_turns": 3,
    }
