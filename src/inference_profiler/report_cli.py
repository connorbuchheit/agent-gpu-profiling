from __future__ import annotations

import argparse
import logging
from pathlib import Path

from inference_profiler.markdown_report import load_bundle, write_markdown_report
from inference_profiler.utils import setup_logging

LOG = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    p = argparse.ArgumentParser(description="Generate Markdown report from metrics.json")
    p.add_argument("--results", required=True, help="Path to metrics.json or run directory")
    p.add_argument("-o", "--output", default=None, help="Output REPORT.md path")
    args = p.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    target = Path(args.results)
    if not target.is_absolute():
        target = root / target
    bundle_path = target if target.is_file() else target / "metrics.json"
    if not bundle_path.is_file():
        raise SystemExit(f"Not found: {bundle_path}")
    load_bundle(bundle_path)  # validate
    out = Path(args.output) if args.output else None
    md = write_markdown_report(bundle_path, out_md=out)
    LOG.info("Wrote %s", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
