"""GPU (and optional CPU) metrics sampling during task runs."""

from .gpu import GPUMetricsCollector, sample_gpu_once

__all__ = ["GPUMetricsCollector", "sample_gpu_once"]
