from __future__ import annotations

"""
PyTorch profiler / Chrome trace integration.

vLLM and SGLang run inference inside a separate process. ``torch.profiler`` only
sees kernels launched in the *current* Python process, so capturing a trace that
includes vLLM's CUDA work requires either:

- running a small in-process forward (this module's ``profile_dummy_cuda_region``), or
- instrumenting the server build (out of scope here).

Use this module for optional local CUDA profiling or as a template when you add
hooks inside a custom inference path.
"""

import logging
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)


def profile_dummy_cuda_region(trace_path: Path, steps: int = 5) -> bool:
    """Run a minimal CUDA matmul under ``torch.profiler`` and export Chrome trace."""
    try:
        import torch
        import torch.profiler as torch_profiler
    except ImportError:
        LOG.warning("PyTorch not available; skip profiler trace")
        return False
    if not torch.cuda.is_available():
        LOG.warning("CUDA not available; skip profiler trace")
        return False

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.randn(4096, 4096, device="cuda")
    y = torch.randn(4096, 4096, device="cuda")

    with torch_profiler.profile(
        activities=[
            torch_profiler.ProfilerActivity.CPU,
            torch_profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(steps):
            z = x @ y
            _ = z.sum().item()

    prof.export_chrome_trace(str(trace_path))
    LOG.info("Wrote Chrome trace to %s", trace_path)
    return True


def print_cuda_top_ops(trace_or_prof: Any, row_limit: int = 15) -> None:
    """If ``trace_or_prof`` is a finished torch.profiler profile, print key averages."""
    try:
        print(trace_or_prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit))
    except Exception as e:
        LOG.debug("Could not print profiler table: %s", e)
