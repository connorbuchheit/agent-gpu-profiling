from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoadTestSettings:
    input_tokens: list[int]
    output_tokens: list[int]
    duration_sec: float | None
    max_requests: int | None
    request_timeout_sec: float
    warmup_requests: int
    random_seed: int


@dataclass
class CostSettings:
    gpu_type: str
    gpu_hourly_usd: float
    theoretical_mem_bw_gbps: float | None


@dataclass
class ServerSettings:
    startup_timeout_sec: float
    health_interval_sec: float
    gpu_memory_utilization: float
    extra_args: list[str]


@dataclass
class RunConfig:
    """One serving configuration + load shape (clients) to benchmark."""

    run_index: int
    model: str
    backend: str
    host: str
    port: int
    batch_size: int
    quantization: str
    max_seq_len: int
    tensor_parallel: int
    kv_cache_strategy: str
    concurrent_clients: int
    load_test: LoadTestSettings
    cost: CostSettings
    server: ServerSettings
    torch_profiler_sample_runs: int
    meta: dict[str, Any] = field(default_factory=dict)

    def run_label(self) -> str:
        return (
            f"b{self.batch_size}_q{self.quantization}_ctx{self.max_seq_len}"
            f"_tp{self.tensor_parallel}_kv{self.kv_cache_strategy}_cc{self.concurrent_clients}"
        )

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def load_matrix_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_runs(doc: dict[str, Any], cli_overrides: dict[str, Any] | None = None) -> list[RunConfig]:
    """Build RunConfig list from YAML document and optional CLI filters."""
    cli_overrides = cli_overrides or {}
    model = str(cli_overrides.get("model") or doc["model"])
    backend = str(doc.get("backend", "vllm")).lower()
    host = str(doc.get("host", "127.0.0.1"))
    base_port = int(doc.get("base_port", 8000))

    m = doc["matrix"]
    keys = [
        ("batch_sizes", "batch_size"),
        ("quantizations", "quantization"),
        ("max_sequence_lengths", "max_seq_len"),
        ("tensor_parallel", "tensor_parallel"),
        ("kv_cache_strategies", "kv_cache_strategy"),
        ("concurrent_clients", "concurrent_clients"),
    ]
    lists_raw = {alias: list(m[src]) for src, alias in keys}

    # CLI single-value filters
    if cli_overrides.get("batch_size") is not None:
        lists_raw["batch_size"] = [int(cli_overrides["batch_size"])]
    if cli_overrides.get("quantization") is not None:
        q = str(cli_overrides["quantization"]).lower()
        lists_raw["quantization"] = [q]
    if cli_overrides.get("max_seq_len") is not None:
        lists_raw["max_seq_len"] = [int(cli_overrides["max_seq_len"])]
    if cli_overrides.get("tensor_parallel") is not None:
        lists_raw["tensor_parallel"] = [int(cli_overrides["tensor_parallel"])]
    if cli_overrides.get("kv_cache_strategy") is not None:
        lists_raw["kv_cache_strategy"] = [str(cli_overrides["kv_cache_strategy"])]
    if cli_overrides.get("concurrent_clients") is not None:
        lists_raw["concurrent_clients"] = [int(cli_overrides["concurrent_clients"])]

    lt = doc["load_test"]
    load_test = LoadTestSettings(
        input_tokens=list(lt["input_tokens"]),
        output_tokens=list(lt["output_tokens"]),
        duration_sec=lt.get("duration_sec"),
        max_requests=lt.get("max_requests"),
        request_timeout_sec=float(lt.get("request_timeout_sec", 120)),
        warmup_requests=int(lt.get("warmup_requests", 0)),
        random_seed=int(lt.get("random_seed", 0)),
    )

    cost_doc = doc.get("cost", {})
    cost = CostSettings(
        gpu_type=str(cost_doc.get("gpu_type", "unknown")),
        gpu_hourly_usd=float(cost_doc.get("gpu_hourly_usd", 0.0)),
        theoretical_mem_bw_gbps=(
            float(cost_doc["theoretical_mem_bw_gbps"])
            if cost_doc.get("theoretical_mem_bw_gbps") is not None
            else None
        ),
    )

    srv = doc.get("server", {})
    server = ServerSettings(
        startup_timeout_sec=float(srv.get("startup_timeout_sec", 300)),
        health_interval_sec=float(srv.get("health_interval_sec", 2.0)),
        gpu_memory_utilization=float(srv.get("gpu_memory_utilization", 0.9)),
        extra_args=list(srv.get("extra_args") or []),
    )

    torch_n = int(doc.get("torch_profiler_sample_runs", 0))

    order = [
        "batch_size",
        "quantization",
        "max_seq_len",
        "tensor_parallel",
        "kv_cache_strategy",
        "concurrent_clients",
    ]
    combos = list(
        itertools.product(
            lists_raw["batch_size"],
            lists_raw["quantization"],
            lists_raw["max_seq_len"],
            lists_raw["tensor_parallel"],
            lists_raw["kv_cache_strategy"],
            lists_raw["concurrent_clients"],
        )
    )

    max_runs = doc.get("max_runs")
    if max_runs is not None:
        combos = combos[: int(max_runs)]

    runs: list[RunConfig] = []
    for i, tup in enumerate(combos):
        (
            batch_size,
            quantization,
            max_seq_len,
            tensor_parallel,
            kv_cache_strategy,
            concurrent_clients,
        ) = tup
        runs.append(
            RunConfig(
                run_index=i,
                model=model,
                backend=backend,
                host=host,
                port=base_port,
                batch_size=int(batch_size),
                quantization=str(quantization).lower(),
                max_seq_len=int(max_seq_len),
                tensor_parallel=int(tensor_parallel),
                kv_cache_strategy=str(kv_cache_strategy),
                concurrent_clients=int(concurrent_clients),
                load_test=load_test,
                cost=cost,
                server=server,
                torch_profiler_sample_runs=torch_n,
                meta={"matrix_path": str(cli_overrides.get("_matrix_path", ""))},
            )
        )
    return runs
