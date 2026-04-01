from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference_profiler.config_matrix import RunConfig, expand_runs, load_matrix_yaml
from inference_profiler.loadtest import run_loadtest, run_warmup
from inference_profiler.metrics import (
    GPUSamplingThread,
    aggregate_requests,
    nvml_gpu_count,
    summarize_gpu_samples,
)
from inference_profiler.pareto import find_pareto_indices, recommend_configs
from inference_profiler.profiler import profile_dummy_cuda_region
from inference_profiler.server import ManagedServer, wait_for_openai_health
from inference_profiler.utils import ensure_dir, sanitize_filename, setup_logging, utc_run_dir_name, write_json
from inference_profiler.visualize import build_dashboard

LOG = logging.getLogger(__name__)


def _serialize_run_config(cfg: RunConfig) -> dict[str, Any]:
    d = asdict(cfg)
    d["run_label"] = cfg.run_label()
    d["base_url"] = cfg.base_url()
    return d


async def _benchmark_one(cfg: RunConfig) -> tuple[list[Any], float]:
    await run_warmup(cfg)
    t0 = time.perf_counter()
    records = await run_loadtest(cfg, discard_results=False)
    wall = time.perf_counter() - t0
    return records, wall


def run_benchmark_suite(
    matrix_path: Path,
    output_root: Path,
    cli_filters: dict[str, Any],
    skip_server: bool,
    dry_run: bool,
    profile_runs: int,
) -> Path:
    doc = load_matrix_yaml(matrix_path)
    cli_filters["_matrix_path"] = str(matrix_path)
    runs = expand_runs(doc, cli_filters)
    LOG.info("Planned runs: %d", len(runs))

    if dry_run:
        for r in runs:
            print(r.run_label())
        return output_root

    run_dir = ensure_dir(output_root / utc_run_dir_name())
    traces_dir = ensure_dir(run_dir / "traces")
    LOG.info("Output directory: %s", run_dir)

    gpus = nvml_gpu_count()
    bundle: dict[str, Any] = {
        "matrix_path": str(matrix_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_count_nvml": gpus,
        "runs": [],
    }

    profile_budget = int(doc.get("torch_profiler_sample_runs", 0)) + int(profile_runs)

    for cfg in runs:
        run_entry: dict[str, Any] = {
            "config": _serialize_run_config(cfg),
            "metrics": {},
            "gpu_time_series": [],
            "gpu_summary": {},
            "errors": [],
            "server_log": None,
        }
        label = sanitize_filename(cfg.run_label())
        this_dir = ensure_dir(run_dir / label)
        run_entry["server_log"] = str(this_dir / "server_stderr.log")

        if gpus is not None and cfg.tensor_parallel > gpus:
            run_entry["errors"].append(
                f"Skip: tensor_parallel={cfg.tensor_parallel} exceeds visible GPUs ({gpus})"
            )
            bundle["runs"].append(run_entry)
            continue

        server: ManagedServer | None = None
        if not skip_server:
            server = ManagedServer(cfg)
            try:
                server.start(log_dir=this_dir)
                ok = wait_for_openai_health(
                    cfg.base_url(),
                    timeout_sec=cfg.server.startup_timeout_sec,
                    interval_sec=cfg.server.health_interval_sec,
                )
                if not ok or server.poll_crash() is not None:
                    run_entry["errors"].append("Server failed health check or exited early")
                    bundle["runs"].append(run_entry)
                    server.stop()
                    continue
            except Exception as e:
                run_entry["errors"].append(f"Server start error: {e}")
                bundle["runs"].append(run_entry)
                if server:
                    server.stop()
                continue
        else:
            ok = wait_for_openai_health(
                cfg.base_url(),
                timeout_sec=30.0,
                interval_sec=1.0,
            )
            if not ok:
                run_entry["errors"].append("skip-server: API not healthy at base_url")
                bundle["runs"].append(run_entry)
                continue

        gpu_thread = GPUSamplingThread(interval_sec=0.5, device_index=0)
        gpu_thread.start()
        try:
            records, wall = asyncio.run(_benchmark_one(cfg))
        except Exception as e:
            run_entry["errors"].append(f"Load test error: {e}")
            records, wall = [], 0.0
        finally:
            gpu_thread.stop_and_join()
            run_entry["gpu_time_series"] = gpu_thread.rows
            run_entry["gpu_summary"] = summarize_gpu_samples(
                gpu_thread.rows,
                cfg.cost.theoretical_mem_bw_gbps,
            )

        run_entry["metrics"] = aggregate_requests(
            records,
            wall_duration_sec=wall,
            cost_hourly_usd=cfg.cost.gpu_hourly_usd,
        )
        run_entry["raw_request_count"] = len(records)

        if profile_budget > 0:
            trace_path = traces_dir / f"{label}_dummy_cuda.json"
            if profile_dummy_cuda_region(trace_path, steps=3):
                run_entry["profiler_chrome_trace"] = str(trace_path)
                profile_budget -= 1

        bundle["runs"].append(run_entry)

        if not skip_server and server is not None:
            server.stop()

    pareto = find_pareto_indices(bundle["runs"])
    bundle["pareto_indices"] = pareto
    for i in pareto:
        if 0 <= i < len(bundle["runs"]):
            bundle["runs"][i]["pareto_optimal"] = True
    bundle["recommendations"] = recommend_configs(bundle["runs"])

    metrics_path = run_dir / "metrics.json"
    write_json(metrics_path, bundle)
    build_dashboard(bundle, run_dir / "dashboard.html")
    LOG.info("Wrote %s", metrics_path)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    p = argparse.ArgumentParser(description="LLM inference configuration matrix benchmark")
    p.add_argument("--config", default="configs/test_matrix.yaml", help="Path to test matrix YAML")
    p.add_argument("--model", default=None, help="Override model id")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--quantization", default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--tensor-parallel", type=int, default=None)
    p.add_argument("--kv-cache-strategy", default=None)
    p.add_argument("--concurrent-clients", type=int, default=None)
    p.add_argument("--results-dir", default="results", help="Root directory for timestamped runs")
    p.add_argument("--skip-server", action="store_true", help="Do not start vLLM; use existing server on matrix host/port")
    p.add_argument("--dry-run", action="store_true", help="Print expanded run labels only")
    p.add_argument(
        "--torch-profiler-samples",
        type=int,
        default=0,
        help="After load tests, emit this many dummy CUDA Chrome traces (not vLLM internals)",
    )
    args = p.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    matrix = Path(args.config)
    if not matrix.is_absolute():
        matrix = root / matrix
    out_root = Path(args.results_dir)
    if not out_root.is_absolute():
        out_root = root / out_root

    filters = {
        "model": args.model,
        "batch_size": args.batch_size,
        "quantization": args.quantization,
        "max_seq_len": args.max_seq_len,
        "tensor_parallel": args.tensor_parallel,
        "kv_cache_strategy": args.kv_cache_strategy,
        "concurrent_clients": args.concurrent_clients,
    }

    out = run_benchmark_suite(
        matrix_path=matrix,
        output_root=out_root,
        cli_filters=filters,
        skip_server=args.skip_server,
        dry_run=args.dry_run,
        profile_runs=args.torch_profiler_samples,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
