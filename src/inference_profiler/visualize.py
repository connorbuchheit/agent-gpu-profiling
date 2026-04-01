from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOG = logging.getLogger(__name__)


def runs_to_dataframe(bundle: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for i, run in enumerate(bundle.get("runs") or []):
        cfg = run.get("config") or {}
        m = run.get("metrics") or {}
        rows.append(
            {
                "run_index": cfg.get("run_index", i),
                "run_label": cfg.get("run_label"),
                "batch_size": cfg.get("batch_size"),
                "quantization": cfg.get("quantization"),
                "max_seq_len": cfg.get("max_seq_len"),
                "tensor_parallel": cfg.get("tensor_parallel"),
                "kv_cache_strategy": cfg.get("kv_cache_strategy"),
                "concurrent_clients": cfg.get("concurrent_clients"),
                "ttft_p99_ms": m.get("ttft_p99_ms"),
                "inter_token_p99_ms": m.get("inter_token_p99_ms"),
                "latency_composite_p99_ms": (m.get("ttft_p99_ms") or 0)
                + (m.get("inter_token_p99_ms") or 0),
                "throughput_tok_s": m.get("throughput_tokens_per_sec_est"),
                "cost_per_million": m.get("cost_per_million_output_tokens_usd"),
                "success_rate": m.get("success_rate"),
                "pareto_optimal": bool(run.get("pareto_optimal")),
            }
        )
    return pd.DataFrame(rows)


def build_dashboard(bundle: dict[str, Any], out_html: Path) -> None:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        LOG.warning("Plotly not installed; skip dashboard: %s", e)
        return

    df = runs_to_dataframe(bundle)
    if df.empty:
        LOG.warning("No runs to plot")
        return

    pareto_idx = set(bundle.get("pareto_indices") or [])

    fig3d = px.scatter_3d(
        df.dropna(subset=["latency_composite_p99_ms", "throughput_tok_s", "cost_per_million"]),
        x="latency_composite_p99_ms",
        y="throughput_tok_s",
        z="cost_per_million",
        color="success_rate",
        hover_data=["run_label", "batch_size", "quantization", "concurrent_clients"],
        title="Latency (composite P99 ms) vs throughput vs cost",
    )
    fig3d.update_layout(scene=dict(xaxis_title="P99 latency (ms)", yaxis_title="tok/s", zaxis_title="$/M out"))

    heat = None
    sub = df.dropna(subset=["batch_size", "quantization", "ttft_p99_ms"])
    if not sub.empty:
        pivot = sub.pivot_table(
            index="batch_size",
            columns="quantization",
            values="ttft_p99_ms",
            aggfunc="mean",
        )
        heat = px.imshow(
            pivot,
            title="Mean P99 TTFT (ms): batch size × quantization",
            aspect="auto",
            color_continuous_scale="Viridis",
        )

    topn = df.sort_values("throughput_tok_s", ascending=False).head(5)
    bar = px.bar(
        topn,
        x="run_label",
        y="throughput_tok_s",
        title="Top 5 runs by throughput (est.)",
    )

    if "pareto_optimal" in df.columns and bool(df["pareto_optimal"].any()):
        pareto_df = df[df["pareto_optimal"]]
    elif pareto_idx:
        pareto_df = df[df["run_index"].isin(pareto_idx)]
    else:
        pareto_df = df.iloc[0:0]
    sc2 = px.scatter(
        df,
        x="throughput_tok_s",
        y="latency_composite_p99_ms",
        color="cost_per_million",
        hover_data=["run_label"],
        title="Throughput vs latency (color=cost)",
    )
    if not pareto_df.empty:
        sc2.add_trace(
            go.Scatter(
                x=pareto_df["throughput_tok_s"],
                y=pareto_df["latency_composite_p99_ms"],
                mode="markers",
                marker=dict(size=14, symbol="star", color="red"),
                name="Pareto",
            )
        )

    parts = [fig3d.to_html(full_html=False, include_plotlyjs="cdn"), sc2.to_html(full_html=False, include_plotlyjs=False)]
    if heat is not None:
        parts.append(heat.to_html(full_html=False, include_plotlyjs=False))
    parts.append(bar.to_html(full_html=False, include_plotlyjs=False))

    html = (
        "<html><head><meta charset='utf-8'><title>Inference profiler dashboard</title></head><body>"
        + "<h1>Inference profiler</h1>"
        + "".join(f"<div style='margin-bottom:2em'>{p}</div>" for p in parts)
        + "</body></html>"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
