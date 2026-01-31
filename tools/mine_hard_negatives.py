#!/usr/bin/env python3
"""
Mine hard-negative frames from videos using timeline CSVs.

Features:
- Extract specific frames (comma-separated list) OR auto-pick high-score frames to review as potential false positives.
- Writes JPEG frames into data/02_processed/hard_negatives/<category>/.
- Appends entries to data/02_processed/hard_negatives/manifest.csv.

Usage examples:

# Manually pick frames 120,240 from a run
python tools/mine_hard_negatives.py \\
  --video results/boston_workzone_short_annotated_fusion.mp4 \\
  --csv results/boston_workzone_short_timeline_fusion.csv \\
  --category orange_trucks \\
  --frames 120,240

# Auto-pick top 40 high-score frames (fused_score_ema >= 0.65) where Phase 1.1 passed
python tools/mine_hard_negatives.py \\
  --video results/boston_workzone_short_annotated_fusion.mp4 \\
  --csv results/boston_workzone_short_timeline_fusion.csv \\
  --category random_cones \\
  --auto-high-score 0.65 \\
  --require-p1 \\
  --max-frames 40

Notes:
- Auto mode is for surfacing likely false positives; you still need to confirm and keep only true negatives.
- For pure negatives with empty labels, keep just the JPEGs; if you add a background class, adjust your YOLO labels accordingly.
"""

import argparse
import csv
from pathlib import Path
from typing import List, Optional

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HARD_NEG_ROOT = ROOT / "data" / "02_processed" / "hard_negatives"
MANIFEST_PATH = HARD_NEG_ROOT / "manifest.csv"


def parse_frame_list(frame_str: str) -> List[int]:
    parts = [p.strip() for p in frame_str.split(",") if p.strip()]
    return [int(p) for p in parts]


def ensure_category_dir(category: str) -> Path:
    out_dir = HARD_NEG_ROOT / category
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def append_manifest(rows: List[dict]):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = MANIFEST_PATH.exists()
    with MANIFEST_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "category", "source", "notes"])
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def extract_frames(
    video_path: Path,
    frame_indices: List[int],
    out_dir: Path,
    category: str,
    source: str,
    notes: str = "",
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    extracted = 0
    manifest_rows = []

    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        fname = f"{video_path.stem}_f{fi}.jpg"
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), frame)
        extracted += 1
        manifest_rows.append(
            {
                "path": str(out_path.relative_to(ROOT)),
                "category": category,
                "source": source,
                "notes": notes,
            }
        )

    append_manifest(manifest_rows)
    cap.release()
    return extracted


def auto_pick_frames(
    df: pd.DataFrame,
    score_th: float,
    max_frames: int,
    require_p1: bool,
) -> List[int]:
    mask = df["fused_score_ema"] >= score_th
    if require_p1 and "p1_multi_cue_pass" in df.columns:
        mask &= df["p1_multi_cue_pass"] == 1
    subset = df[mask].copy()
    subset = subset.sort_values("fused_score_ema", ascending=False)
    frames = subset["frame"].astype(int).tolist()
    return frames[:max_frames]


def main():
    parser = argparse.ArgumentParser(description="Mine hard-negative frames")
    parser.add_argument("--video", required=True, help="Path to annotated video mp4")
    parser.add_argument("--csv", required=True, help="Path to timeline CSV")
    parser.add_argument("--category", required=True, help="Hard-negative category name")
    parser.add_argument("--frames", default=None, help="Comma-separated frame indices to extract")
    parser.add_argument("--auto-high-score", type=float, default=None, help="Auto-pick frames with fused_score_ema >= TH")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames to extract in auto mode")
    parser.add_argument("--require-p1", action="store_true", help="In auto mode, require p1_multi_cue_pass == 1")
    parser.add_argument("--notes", default="", help="Notes to store in manifest")
    args = parser.parse_args()

    video_path = Path(args.video)
    csv_path = Path(args.csv)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    out_dir = ensure_category_dir(args.category)
    df = pd.read_csv(csv_path)

    frame_indices: List[int] = []

    if args.frames:
        frame_indices.extend(parse_frame_list(args.frames))

    if args.auto_high_score is not None:
        auto_frames = auto_pick_frames(
            df,
            score_th=float(args.auto_high_score),
            max_frames=args.max_frames,
            require_p1=args.require_p1,
        )
        frame_indices.extend(auto_frames)

    # Deduplicate and sort
    frame_indices = sorted(set(frame_indices))

    if not frame_indices:
        raise SystemExit("No frames selected. Provide --frames or --auto-high-score.")

    extracted = extract_frames(
        video_path=video_path,
        frame_indices=frame_indices,
        out_dir=out_dir,
        category=args.category,
        source=str(csv_path.relative_to(ROOT)),
        notes=args.notes,
    )

    print(f"Extracted {extracted} frame(s) to {out_dir}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
