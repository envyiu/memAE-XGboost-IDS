#!/usr/bin/env python
"""CLI: Generate a detailed model evaluation report from a pipeline summary."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.detailed_report import generate_detailed_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a detailed evaluation report with visualizations."
    )
    parser.add_argument(
        "--summary-json", required=True,
        help="Path to full_pipeline_*_summary.json.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Defaults to <summary-dir>/detailed_report.",
    )
    parser.add_argument(
        "--families", nargs="*", default=None,
        help="Restrict to specific families (e.g. web_attack portscan).",
    )
    parser.add_argument(
        "--sample-size", type=int, default=100_000,
        help="Max samples per split for faster plotting.",
    )
    args = parser.parse_args()
    generate_detailed_report(
        summary_json=args.summary_json,
        output_dir=args.output_dir,
        families=args.families,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
