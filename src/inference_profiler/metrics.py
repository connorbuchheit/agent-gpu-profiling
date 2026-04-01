from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np
import psutil

try:
    import pynvml
except ImportError:
    pynvml = None  # type: ignore

from inference_profiler.loadtest import RequestRecord


def _percentiles_ms(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f"{prefix}_p50_ms": float("nan"), f"{prefix}_p90_ms": float("nan"), f"{prefix}_p99_ms": float("nan"), f"{prefix}_max_ms": float("nan")}
    a = np.array(values, dtype=np.float64)
    p50, p90, p99 = np.percentile(a, [50, 90, 99])
    return {
        f"{prefix}_p50_ms": round(float(p50) * 1000, 3),
        f"{prefix}_p90_ms": round(float(p90) * 1000, 3),
        f"{prefix}_p99_ms": round(float(p99) * 1000, 3),
        f"{prefix}_max_ms": round(float(np.max(a)) * 1000, 3),
    }


def flatten_itl(records: list[RequestRecord]) -> list[float]:
    out: list[float] = []
    for r in records:
        if r.ok and r.inter_token_latencies_sec:
            out.extend(r.inter_token_latencies_sec)
    return out


def aggregate_requests(
    records: list[RequestRecord],
    wall_duration_sec: float,
    cost_hourly_usd: float,
) -> dict[str, Any]:
    ok = [r for r in records if r.ok]
    failed = [r for r in records if not r.ok]
    n = len(records)
    success_rate = (len(ok) / n) if n else 0.0

    ttfts = [r.ttft_sec for r in ok if r.ttft_sec is not None]
    totals = [r.total_latency_sec for r in ok]
    itl = flatten_itl(ok)

    out_toks = sum(r.output_tokens_est for r in ok)
    in_toks = sum(r.input_tokens_est for r in ok)

    tok_per_sec = (out_toks / wall_duration_sec) if wall_duration_sec > 0 else 0.0
    req_per_sec = (len(ok) / wall_duration_sec) if wall_duration_sec > 0 else 0.0

    runtime_hours = wall_duration_sec / 3600.0
    cost_usd = runtime_hours * cost_hourly_usd
    cost_per_million_out = (cost_usd / out_toks * 1_000_000) if out_toks > 0 else float("inf")

    m: dict[str, Any] = {
        "n_requests": n,
        "n_success": len(ok),
        "n_failed": len(failed),
        "success_rate": round(success_rate, 4),
        "wall_duration_sec": round(wall_duration_sec, 3),
        "output_tokens_total_est": out_toks,
        "input_tokens_total_est": in_toks,
        "throughput_tokens_per_sec_est": round(tok_per_sec, 2),
        "throughput_requests_per_sec": round(req_per_sec, 4),
        "cost_usd_est": round(cost_usd, 6),
        "cost_per_million_output_tokens_usd": round(cost_per_million_out, 4)
        if math.isfinite(cost_per_million_out)
        else None,
    }
    m.update(_percentiles_ms(ttfts, "ttft"))
    m.update(_percentiles_ms(totals, "total_latency"))
    m.update(_percentiles_ms(itl, "inter_token"))
    if failed:
        errs: dict[str, int] = {}
        for r in failed:
            k = (r.error or "unknown")[:80]
            errs[k] = errs.get(k, 0) + 1
        m["error_histogram"] = errs
    return m


class GPUSamplingThread(threading.Thread):
    """Background NVML + psutil sampling until stop()."""

    def __init__(self, interval_sec: float = 0.5, device_index: int = 0):
        super().__init__(daemon=True)
        self.interval_sec = interval_sec
        self.device_index = device_index
        self._stop = threading.Event()
        self.rows: list[dict[str, Any]] = []

    def run(self) -> None:
        t0 = time.perf_counter()
        while not self._stop.wait(self.interval_sec):
            row: dict[str, Any] = {"elapsed_sec": round(time.perf_counter() - t0, 3)}
            row.update(_sample_psutil_net_cpu())
            row.update(_sample_gpu_nvml(self.device_index))
            self.rows.append(row)

    def stop_and_join(self) -> None:
        self._stop.set()
        self.join(timeout=self.interval_sec * 3)


def _sample_psutil_net_cpu() -> dict[str, Any]:
    try:
        cpu = psutil.cpu_percent(interval=None)
        net = psutil.net_io_counters()
        return {
            "cpu_pct": round(float(cpu), 2),
            "net_bytes_sent": net.bytes_sent,
            "net_bytes_recv": net.bytes_recv,
        }
    except Exception:
        return {"cpu_pct": None, "net_bytes_sent": None, "net_bytes_recv": None}


def _sample_gpu_nvml(index: int) -> dict[str, Any]:
    if pynvml is None:
        return {
            "gpu_util_pct": None,
            "gpu_mem_used_mb": None,
            "gpu_mem_total_mb": None,
            "gpu_temp_c": None,
            "gpu_power_w": None,
            "gpu_mem_bw_util_pct": None,
        }
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_w = round(power_mw / 1000.0, 2)
        except Exception:
            power_w = None
        pynvml.nvmlShutdown()
        return {
            "gpu_util_pct": int(util.gpu),
            "gpu_mem_used_mb": round(mem.used / (1024 * 1024), 2),
            "gpu_mem_total_mb": round(mem.total / (1024 * 1024), 2),
            "gpu_temp_c": int(temp),
            "gpu_power_w": power_w,
            "gpu_mem_bw_util_pct": None,
        }
    except Exception:
        return {
            "gpu_util_pct": None,
            "gpu_mem_used_mb": None,
            "gpu_mem_total_mb": None,
            "gpu_temp_c": None,
            "gpu_power_w": None,
            "gpu_mem_bw_util_pct": None,
        }


def summarize_gpu_samples(
    rows: list[dict[str, Any]],
    theoretical_mem_bw_gbps: float | None,
) -> dict[str, Any]:
    if not rows:
        return {}
    util = [r["gpu_util_pct"] for r in rows if r.get("gpu_util_pct") is not None]
    memu = [r["gpu_mem_used_mb"] for r in rows if r.get("gpu_mem_used_mb") is not None]
    temp = [r["gpu_temp_c"] for r in rows if r.get("gpu_temp_c") is not None]
    pwr = [r["gpu_power_w"] for r in rows if r.get("gpu_power_w") is not None]
    cpu = [r["cpu_pct"] for r in rows if r.get("cpu_pct") is not None]

    def _mean(xs: list[float]) -> float | None:
        return float(np.mean(xs)) if xs else None

    out: dict[str, Any] = {
        "gpu_util_mean": round(_mean(util), 2) if util else None,
        "gpu_util_max": int(np.max(util)) if util else None,
        "gpu_mem_used_mean_mb": round(float(np.mean(memu)), 2) if memu else None,
        "gpu_mem_used_max_mb": round(float(np.max(memu)), 2) if memu else None,
        "gpu_temp_max_c": int(np.max(temp)) if temp else None,
        "gpu_power_mean_w": round(float(np.mean(pwr)), 2) if pwr else None,
        "cpu_mean_pct": round(float(np.mean(cpu)), 2) if cpu else None,
    }
    if theoretical_mem_bw_gbps:
        out["theoretical_mem_bw_gbps"] = theoretical_mem_bw_gbps
    return out


def nvml_gpu_count() -> int | None:
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return int(n)
    except Exception:
        return None
