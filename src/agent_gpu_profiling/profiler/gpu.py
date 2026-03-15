"""Sample GPU utilization and memory at an interval; write results to CSV."""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Any

try:
    import pynvml
except ImportError:
    pynvml = None  # type: ignore


def sample_gpu_once() -> dict[str, Any] | None:
    """Take a single GPU sample (util % and memory). Returns None if unavailable."""
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return {
            "gpu_util_pct": util.gpu,
            "gpu_mem_used_mb": round(mem.used / (1024 * 1024), 2),
            "gpu_mem_total_mb": round(mem.total / (1024 * 1024), 2),
        }
    except Exception:
        return None


class GPUMetricsCollector:
    """Background collector: samples GPU metrics at a fixed interval and writes to CSV."""

    def __init__(
        self,
        output_path: str | Path,
        interval_sec: float = 1.0,
        backend: str = "",
        task_type: str = "",
        run_id: str = "",
    ):
        self.output_path = Path(output_path)
        self.interval_sec = interval_sec
        self.backend = backend
        self.task_type = task_type
        self.run_id = run_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rows: list[dict[str, Any]] = []

    def _sample(self) -> dict[str, Any] | None:
        if pynvml is None:
            return {"gpu_util_pct": None, "gpu_mem_used_mb": None, "gpu_mem_alloc_mb": None}
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return {
                "gpu_util_pct": util.gpu,
                "gpu_mem_used_mb": round(mem.used / (1024 * 1024), 2),
                "gpu_mem_alloc_mb": round(mem.total / (1024 * 1024), 2),
            }
        except Exception:
            return None

    def _run(self):
        start = time.perf_counter()
        while not self._stop.wait(timeout=self.interval_sec):
            row = self._sample()
            if row is not None:
                row["elapsed_sec"] = round(time.perf_counter() - start, 2)
                row["backend"] = self.backend
                row["task_type"] = self.task_type
                row["run_id"] = self.run_id
                self._rows.append(row)

    def start(self):
        """Start background sampling."""
        self._rows = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop sampling and write CSV."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec * 2)
        self._write_csv()

    def _write_csv(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._rows:
            return
        with open(self.output_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            w.writeheader()
            w.writerows(self._rows)
