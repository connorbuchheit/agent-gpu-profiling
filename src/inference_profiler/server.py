from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import urllib.error
import urllib.request

if TYPE_CHECKING:
    from inference_profiler.config_matrix import RunConfig

LOG = logging.getLogger(__name__)


def _quantization_to_vllm_args(quant: str) -> tuple[str, list[str]]:
    """Return (dtype flag value, extra CLI args)."""
    q = quant.lower().strip()
    if q in ("fp16", "float16", "half"):
        return "half", []
    if q in ("bf16", "bfloat16"):
        return "bfloat16", []
    if q in ("fp32", "float32"):
        return "float32", []
    if q in ("fp8", "float8"):
        # vLLM version-dependent; common pattern:
        return "auto", ["--quantization", "fp8"]
    if q in ("gptq", "int8", "int4"):
        return "auto", ["--quantization", "gptq"]
    if q in ("awq",):
        return "auto", ["--quantization", "awq"]
    LOG.warning("Unknown quantization %r; using dtype=auto", quant)
    return "auto", []


def build_vllm_command(cfg: RunConfig) -> list[str]:
    dtype, extra = _quantization_to_vllm_args(cfg.quantization)
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        cfg.model,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--max-model-len",
        str(cfg.max_seq_len),
        "--tensor-parallel-size",
        str(cfg.tensor_parallel),
        "--max-num-seqs",
        str(cfg.batch_size),
        "--dtype",
        dtype,
        "--gpu-memory-utilization",
        str(cfg.server.gpu_memory_utilization),
    ]
    # vLLM uses PagedAttention by default. kv_cache_strategy is recorded in results for comparison
    # across backends or future flags; map differences via server.extra_args if needed.
    cmd.extend(extra)
    cmd.extend(cfg.server.extra_args)
    return cmd


def build_sglang_command(cfg: RunConfig) -> list[str]:
    """Minimal SGLang HTTP server command (install sglang separately)."""
    dtype, _ = _quantization_to_vllm_args(cfg.quantization)
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        cfg.model,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--mem-fraction-static",
        str(cfg.server.gpu_memory_utilization),
    ]
    if dtype != "auto":
        cmd.extend(["--dtype", dtype])
    cmd.extend(cfg.server.extra_args)
    return cmd


def wait_for_openai_health(base_url: str, timeout_sec: float, interval_sec: float) -> bool:
    """Poll GET {base_url}/models until 200 or timeout."""
    url = base_url.rstrip("/") + "/models"
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            LOG.debug("Health check pending: %s", e)
        time.sleep(interval_sec)
    return False


class ManagedServer:
    """Start and stop an inference server subprocess (vLLM or SGLang)."""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.proc: subprocess.Popen[str] | None = None
        self._log_path: Path | None = None
        self._stderr_file: object | None = None

    def start(self, log_dir: Path | None = None) -> None:
        if self.cfg.backend == "sglang":
            cmd = build_sglang_command(self.cfg)
        else:
            cmd = build_vllm_command(self.cfg)
        LOG.info("Starting server: %s", " ".join(cmd))
        env = os.environ.copy()
        stderr: int | object = subprocess.DEVNULL
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = log_dir / "server_stderr.log"
            self._stderr_file = open(self._log_path, "w", encoding="utf-8")
            stderr = self._stderr_file
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            text=True,
            env=env,
            start_new_session=True,
        )

    def stop(self, grace_sec: float = 15.0) -> None:
        if self.proc is None:
            return
        pid = self.proc.pid
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            self.proc.terminate()
        try:
            self.proc.wait(timeout=grace_sec)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                self.proc.kill()
            self.proc.wait(timeout=5)
        self.proc = None
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except OSError:
                pass
            self._stderr_file = None

    def poll_crash(self) -> int | None:
        if self.proc is None:
            return None
        return self.proc.poll()
