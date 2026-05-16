#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from workzone_metrics.report import generate_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tolerance sweep over transition matching.")
    parser.add_argument("--gt", required=True, help="Path to ground-truth JSON.")
    parser.add_argument("--pred", required=True, help="Path to predictions JSON/CSV/dir.")
    parser.add_argument(
        "--tolerances",
        default="0,5,10,15,30,60",
        help="Comma-separated list of frame tolerances to evaluate.",
    )
    parser.add_argument(
        "--out",
        default="results/tolerance_sweep.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tolerances = [int(x.strip()) for x in args.tolerances.split(",") if x.strip()]
    if not tolerances:
        raise ValueError("No tolerances provided.")

    results = {}
    for tol in tolerances:
        report = generate_report(args.gt, args.pred, transition_tolerance_frames=tol)
        results[str(tol)] = report["summary"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
