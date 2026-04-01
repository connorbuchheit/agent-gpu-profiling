from __future__ import annotations

import argparse
import logging
from pathlib import Path

from inference_profiler.markdown_report import load_bundle, write_markdown_report
from inference_profiler.pareto import find_pareto_indices, recommend_configs
from inference_profiler.utils import setup_logging, write_json
from inference_profiler.visualize import build_dashboard

LOG = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    p = argparse.ArgumentParser(description="Analyze existing benchmark results")
    p.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing metrics.json (or path to metrics.json)",
    )
    p.add_argument("--replot", action="store_true", help="Regenerate dashboard.html from metrics.json")
    p.add_argument("--report", action="store_true", help="Write REPORT.md next to metrics.json")
    args = p.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    target = Path(args.results_dir)
    if not target.is_absolute():
        target = root / target

    bundle = load_bundle(target)
    metrics_path = target if target.is_file() else target / "metrics.json"

    pareto = find_pareto_indices(bundle.get("runs") or [])
    bundle["pareto_indices"] = pareto
    bundle["recommendations"] = recommend_configs(bundle.get("runs") or [])
    for i, run in enumerate(bundle.get("runs") or []):
        run["pareto_optimal"] = i in pareto

    write_json(metrics_path, bundle)
    LOG.info("Pareto indices: %s", pareto)
    LOG.info("Recommendations: %s", bundle["recommendations"])

    if args.replot:
        build_dashboard(bundle, metrics_path.parent / "dashboard.html")
        LOG.info("Updated dashboard at %s", metrics_path.parent / "dashboard.html")

    if args.report:
        md = write_markdown_report(metrics_path)
        LOG.info("Wrote %s", md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
