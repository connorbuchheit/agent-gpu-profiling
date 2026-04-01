#!/usr/bin/env python3
"""LLM inference matrix benchmark (vLLM/SGLang). PYTHONPATH must include ./src or install editable."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "src"))

from inference_profiler.benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
