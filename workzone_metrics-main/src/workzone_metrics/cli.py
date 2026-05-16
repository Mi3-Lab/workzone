import argparse

from .report import generate_report, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate workzone metrics.")
    parser.add_argument("--gt", required=True, help="Path to ground-truth JSON.")
    parser.add_argument(
        "--pred",
        required=True,
        help="Path to predictions JSON, a timeline CSV, or a directory of timeline CSVs.",
    )
    parser.add_argument("--out", help="Optional path to write the report JSON.")
    parser.add_argument(
        "--transition-tolerance-frames",
        type=int,
        default=0,
        help="Allowed frame tolerance when matching state transitions.",
    )
    parser.add_argument(
        "--min-event-overlap-frames",
        type=int,
        default=1,
        help="Minimum overlap (frames) to match GT/pred INSIDE events.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = generate_report(
        args.gt,
        args.pred,
        transition_tolerance_frames=args.transition_tolerance_frames,
        min_event_overlap_frames=args.min_event_overlap_frames,
    )
    write_report(report, args.out)


if __name__ == "__main__":
    main()
