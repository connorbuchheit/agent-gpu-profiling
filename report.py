#!/usr/bin/env python3
"""Generate REPORT.md from metrics.json."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "src"))

from inference_profiler.report_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
