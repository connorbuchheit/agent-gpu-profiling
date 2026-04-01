from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import aiohttp

from inference_profiler.utils import make_prompt

if TYPE_CHECKING:
    from inference_profiler.config_matrix import RunConfig

LOG = logging.getLogger(__name__)


@dataclass
class RequestRecord:
    ok: bool
    error: str | None
    total_latency_sec: float
    ttft_sec: float | None
    inter_token_latencies_sec: list[float] = field(default_factory=list)
    output_tokens_est: int
    input_tokens_est: int


def _parse_sse_line(line: bytes) -> dict[str, Any] | None:
    if not line.startswith(b"data:"):
        return None
    rest = line[5:].strip()
    if rest == b"[DONE]":
        return {"done": True}
    try:
        return json.loads(rest.decode("utf-8"))
    except json.JSONDecodeError:
        return None


async def _one_streaming_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    timeout_sec: float,
) -> RequestRecord:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft: float | None = None
    itl: list[float] = []
    last_chunk_t = t0
    out_chars = 0
    in_est = max(1, (len(system_prompt) + len(user_prompt)) // 4)

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_sec, sock_read=timeout_sec)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestRecord(
                    ok=False,
                    error=f"HTTP {resp.status}: {body[:400]}",
                    total_latency_sec=time.perf_counter() - t0,
                    ttft_sec=None,
                    inter_token_latencies_sec=[],
                    output_tokens_est=0,
                    input_tokens_est=in_est,
                )
            async for line in resp.content:
                if not line:
                    continue
                for part in line.split(b"\n"):
                    part = part.strip()
                    if not part:
                        continue
                    data = _parse_sse_line(part)
                    if data is None or data.get("done"):
                        continue
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0] or {}).get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        else:
                            itl.append(now - last_chunk_t)
                        last_chunk_t = now
                        out_chars += len(piece)
        total = time.perf_counter() - t0
        out_tok = max(1, out_chars // 4)
        return RequestRecord(
            ok=True,
            error=None,
            total_latency_sec=total,
            ttft_sec=ttft,
            inter_token_latencies_sec=itl,
            output_tokens_est=out_tok,
            input_tokens_est=in_est,
        )
    except asyncio.TimeoutError:
        return RequestRecord(
            ok=False,
            error="timeout",
            total_latency_sec=time.perf_counter() - t0,
            ttft_sec=ttft,
            inter_token_latencies_sec=itl,
            output_tokens_est=max(1, out_chars // 4),
            input_tokens_est=in_est,
        )
    except aiohttp.ClientError as e:
        return RequestRecord(
            ok=False,
            error=str(e)[:300],
            total_latency_sec=time.perf_counter() - t0,
            ttft_sec=ttft,
            inter_token_latencies_sec=itl,
            output_tokens_est=max(1, out_chars // 4),
            input_tokens_est=in_est,
        )


async def run_loadtest(cfg: RunConfig, discard_results: bool = False) -> list[RequestRecord]:
    """Concurrent load until duration_sec and/or max_requests (whichever triggers first)."""
    lt = cfg.load_test
    rng_master = random.Random(lt.random_seed + cfg.run_index)
    base = cfg.base_url()

    max_requests = lt.max_requests
    duration_sec = lt.duration_sec
    if duration_sec is None and max_requests is None:
        duration_sec = 60.0

    deadline = time.monotonic() + float(duration_sec) if duration_sec is not None else None
    lock = asyncio.Lock()
    records: list[RequestRecord] = []

    def _done_by_time() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    async def worker(worker_id: int) -> None:
        rng = random.Random(rng_master.randint(0, 2**30) + worker_id)
        while True:
            if _done_by_time():
                return
            async with lock:
                if max_requests is not None and len(records) >= max_requests:
                    return
            in_toks = rng.choice(lt.input_tokens)
            out_toks = rng.choice(lt.output_tokens)
            user = make_prompt(in_toks, seed=rng.randint(0, 2**30))
            system = make_prompt(max(50, in_toks // 10), seed=rng.randint(0, 2**30))
            # session created outside
            rec = await _one_streaming_request(
                session,
                base,
                cfg.model,
                system,
                user,
                max_tokens=out_toks,
                timeout_sec=lt.request_timeout_sec,
            )
            if discard_results:
                continue
            async with lock:
                records.append(rec)
                if max_requests is not None and len(records) >= max_requests:
                    return

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(worker(i)) for i in range(cfg.concurrent_clients)
        ]
        try:
            if deadline is not None:
                while time.monotonic() < deadline:
                    async with lock:
                        if max_requests is not None and len(records) >= max_requests:
                            break
                    await asyncio.sleep(0.15)
            else:
                await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    return records


async def run_warmup(cfg: RunConfig) -> None:
    """Sequential warmup requests (discarded)."""
    n = cfg.load_test.warmup_requests
    if n <= 0:
        return
    lt = cfg.load_test
    rng = random.Random(lt.random_seed)
    base = cfg.base_url()
    connector = aiohttp.TCPConnector(limit=4)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(n):
            in_toks = rng.choice(lt.input_tokens)
            out_toks = min(rng.choice(lt.output_tokens), 64)
            user = make_prompt(in_toks, seed=rng.randint(0, 2**30))
            system = make_prompt(80, seed=i)
            await _one_streaming_request(
                session,
                base,
                cfg.model,
                system,
                user,
                max_tokens=out_toks,
                timeout_sec=min(lt.request_timeout_sec, 120.0),
            )
