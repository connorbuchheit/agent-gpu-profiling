from __future__ import annotations

from typing import Any


def _objective_vector(row: dict[str, Any]) -> tuple[float, float, float] | None:
    """Return (latency_ms, neg_throughput, cost) for minimization; None if incomplete."""
    m = row.get("metrics") or {}
    lat = m.get("ttft_p99_ms")
    itl = m.get("inter_token_p99_ms")
    tput = m.get("throughput_tokens_per_sec_est")
    cost = m.get("cost_per_million_output_tokens_usd")
    if lat is None or itl is None or tput is None or cost is None:
        return None
    try:
        if float(tput) <= 0:
            return None
        latency = float(lat) + float(itl)
        return (latency, -float(tput), float(cost))
    except (TypeError, ValueError):
        return None


def find_pareto_indices(rows: list[dict[str, Any]]) -> list[int]:
    """Indices of runs whose (TTFT+ITL P99, -tokens/s, $/M) vector is non-dominated."""
    indexed: list[tuple[int, tuple[float, float, float]]] = []
    for i, row in enumerate(rows):
        v = _objective_vector(row)
        if v is not None:
            indexed.append((i, v))
    pareto: list[int] = []
    for i, v in indexed:
        dominated = False
        for j, u in indexed:
            if i == j:
                continue
            if u[0] <= v[0] and u[1] <= v[1] and u[2] <= v[2]:
                if u[0] < v[0] or u[1] < v[1] or u[2] < v[2]:
                    dominated = True
                    break
        if not dominated:
            pareto.append(i)
    return pareto


def recommend_configs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Heuristic picks for common deployment goals."""
    valid: list[tuple[int, dict[str, Any], float, float, float]] = []
    for i, row in enumerate(rows):
        m = row.get("metrics") or {}
        if m.get("success_rate", 0) < 0.5:
            continue
        tput = m.get("throughput_tokens_per_sec_est")
        cost = m.get("cost_per_million_output_tokens_usd")
        lat = (m.get("ttft_p99_ms") or 0) + (m.get("inter_token_p99_ms") or 0)
        if tput is None or cost is None:
            continue
        valid.append((i, row, float(lat), float(tput), float(cost)))

    out: dict[str, Any] = {}
    if not valid:
        return out

    under_100 = [x for x in valid if x[2] <= 100]
    if under_100:
        best = max(under_100, key=lambda x: x[3])
        out["lowest_latency_under_100ms_p99_composite"] = _run_label(best[1])

    under_cost = [x for x in valid if x[4] <= 10.0]
    if under_cost:
        best_t = max(under_cost, key=lambda x: x[3])
        out["best_throughput_under_10usd_per_million"] = _run_label(best_t[1])

    lats = [x[2] for x in valid]
    tputs = [x[3] for x in valid]
    costs = [x[4] for x in valid]
    l_min, l_max = min(lats), max(lats)
    t_min, t_max = min(tputs), max(tputs)
    c_min, c_max = min(costs), max(costs)

    def norm(x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return (x - lo) / (hi - lo)

    scored: list[tuple[float, dict[str, Any]]] = []
    for _i, row, lat, tput, cost in valid:
        score = (
            norm(lat, l_min, l_max)
            + norm(cost, c_min, c_max)
            + (1.0 - norm(tput, t_min, t_max))
        )
        scored.append((score, row))
    scored.sort(key=lambda z: z[0])
    out["balanced_production_heuristic"] = _run_label(scored[0][1])
    return out


def _run_label(row: dict[str, Any]) -> str:
    c = row.get("config") or {}
    return str(c.get("run_label") or c.get("id") or "unknown")
