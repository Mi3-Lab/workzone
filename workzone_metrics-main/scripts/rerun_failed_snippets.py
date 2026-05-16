import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun failed workzone snippets.")
    parser.add_argument("--gt", required=True, help="Path to ground-truth JSON.")
    parser.add_argument(
        "--videos",
        required=True,
        help="Directory containing *_snippet.mp4 files.",
    )
    parser.add_argument(
        "--outputs",
        required=True,
        help="Batch outputs directory (per-video subfolders).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Frame stride (default 2).",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip annotated video output.",
    )
    parser.add_argument(
        "--workzone-root",
        default=None,
        help="Path to workzone repo root (defaults to ./workzone-main/workzone-main).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of videos to rerun.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.gt)
    videos_dir = Path(args.videos)
    outputs_dir = Path(args.outputs)

    if args.workzone_root:
        workzone_root = Path(args.workzone_root)
    else:
        workzone_root = Path(__file__).resolve().parents[1] / "workzone-main" / "workzone-main"

    process_script = workzone_root / "scripts" / "process_video_fusion.py"
    if not process_script.exists():
        raise SystemExit(f"process_video_fusion.py not found at {process_script}")

    with gt_path.open("r", encoding="utf-8") as f:
        gt = json.load(f)

    missing: List[str] = []
    for name in sorted(gt.keys()):
        stem = name.replace(".mp4", "")
        out_dir = outputs_dir / stem
        out_csv = out_dir / f"{stem}_timeline_fusion.csv"
        if not out_csv.exists():
            missing.append(name)

    if args.limit is not None:
        missing = missing[: args.limit]

    outputs_dir.mkdir(parents=True, exist_ok=True)
    failed_path = outputs_dir / "failed_videos.txt"
    failed_path.write_text("\n".join(missing), encoding="utf-8")
    print(f"Missing timelines: {len(missing)}")
    if not missing:
        return

    tmp_dir = Path(os.environ.get("TMPDIR", ""))
    if not tmp_dir:
        tmp_dir = Path.cwd() / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "offline"

    for idx, name in enumerate(missing, start=1):
        video_path = (videos_dir / name).resolve()
        if not video_path.exists():
            print(f"SKIP missing file: {name}")
            continue
        stem = name.replace(".mp4", "")
        out_dir = (outputs_dir / stem).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(Path.cwd() / ".venv" / "bin" / "python"),
            str(process_script),
            str(video_path),
            "--output-dir",
            str(out_dir),
            "--device",
            args.device,
            "--stride",
            str(args.stride),
        ]
        if args.no_video:
            cmd.append("--no-video")

        print(f"[{idx}/{len(missing)}] {name}")
        result = subprocess.run(cmd, cwd=str(workzone_root), env=env)
        if result.returncode != 0:
            print(f"FAILED {name}")


if __name__ == "__main__":
    main()
