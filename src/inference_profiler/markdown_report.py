from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from inference_profiler.utils import read_json
from inference_profiler.visualize import runs_to_dataframe


def write_markdown_report(bundle_path: Path, out_md: Path | None = None) -> Path:
    bundle = read_json(bundle_path)
    if out_md is None:
        out_md = bundle_path.parent / "REPORT.md"
    df = runs_to_dataframe(bundle)
    lines: list[str] = []
    lines.append("# Inference benchmark report\n")
    lines.append(f"- Matrix: `{bundle.get('matrix_path')}`\n")
    lines.append(f"- Created (UTC): {bundle.get('created_utc')}\n")

    rec = bundle.get("recommendations") or {}
    if rec:
        lines.append("\n## Recommendations\n")
        for k, v in rec.items():
            lines.append(f"- **{k}**: `{v}`\n")

    pareto = bundle.get("pareto_indices") or []
    if pareto:
        lines.append("\n## Pareto-optimal runs (index)\n")
        lines.append(", ".join(str(i) for i in pareto))
        lines.append("\n")

    if not df.empty:
        lines.append("\n## Top 5 by P99 TTFT (ms)\n")
        sub = df.dropna(subset=["ttft_p99_ms"]).sort_values("ttft_p99_ms").head(5)
        lines.append(sub.to_markdown(index=False))
        lines.append("\n")

        lines.append("\n## Top 5 by throughput (tok/s est.)\n")
        sub2 = df.dropna(subset=["throughput_tok_s"]).sort_values("throughput_tok_s", ascending=False).head(5)
        lines.append(sub2.to_markdown(index=False))
        lines.append("\n")

        lines.append("\n## Top 5 by cost ($/M output tokens)\n")
        sub3 = df.dropna(subset=["cost_per_million"]).sort_values("cost_per_million").head(5)
        lines.append(sub3.to_markdown(index=False))
        lines.append("\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def load_bundle(path_or_dir: Path) -> dict[str, Any]:
    p = path_or_dir
    if p.is_dir():
        cand = p / "metrics.json"
        if not cand.is_file():
            raise FileNotFoundError(f"No metrics.json under {p}")
        p = cand
    return read_json(p)
