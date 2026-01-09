#!/usr/bin/env python3
"""
Generate calibration images for TensorRT INT8 calibration by sampling frames
from dataset videos (data/demo or data/videos_compressed).

Usage:
    python scripts/generate_calib_images.py --src data/demo --out calib/int8_calib --num-images 500 --imgsz 1080
"""

import argparse
from pathlib import Path
import random
import cv2
import math


def find_videos(src: Path):
    video_exts = ("*.mp4", "*.mov", "*.avi", "*.mkv")
    vids = []
    for e in video_exts:
        vids.extend(sorted(src.glob(e)))
    return vids


def get_frame_count(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, cnt)


def extract_frames_from_video(video_path: Path, out_dir: Path, indices, imgsz: int, start_idx: int = 0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️  Could not open video: {video_path}")
        return 0

    saved = 0
    indices_set = set(indices)
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices_set:
            # Resize to square imgsz x imgsz (center-crop then resize to preserve content)
            h, w = frame.shape[:2]
            # center crop to square
            if h > w:
                start = (h - w) // 2
                crop = frame[start:start + w, :, :]
            else:
                start = (w - h) // 2
                crop = frame[:, start:start + h, :]
            resized = cv2.resize(crop, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
            out_path = out_dir / f"{start_idx + saved:06d}.jpg"
            cv2.imwrite(str(out_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved += 1
            if saved >= len(indices):
                break
        idx += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="data/demo", help="Source video directory")
    parser.add_argument("--out", type=str, default="calib/int8_calib", help="Output image directory")
    parser.add_argument("--num-images", type=int, default=500, help="Number of calibration images to create")
    parser.add_argument("--imgsz", type=int, default=1080, help="Output image size (square) in pixels, e.g., 1080")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--min-per-video", type=int, default=2, help="Minimum images to sample from each video")

    args = parser.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out)

    if not src.exists():
        print(f"❌ Source directory does not exist: {src}")
        return

    vids = find_videos(src)
    if not vids:
        print(f"❌ No video files found in: {src}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute frames per video proportionally
    counts = [get_frame_count(v) for v in vids]
    total_frames = sum(counts)
    if total_frames == 0:
        print("❌ No frames found in videos")
        return

    random.seed(args.seed)

    # Allocate target counts per video
    target = []
    remaining = args.num_images
    for i, c in enumerate(counts):
        if i == len(counts) - 1:
            n = remaining
        else:
            # proportional allocation with minimum
            n = max(args.min_per_video, int(round(c / total_frames * args.num_images)))
            remaining -= n
        target.append(n)

    # If overallocation or underallocation, fix by adjusting last
    allocated = sum(target)
    if allocated != args.num_images:
        diff = args.num_images - allocated
        target[-1] += diff

    print(f"Found {len(vids)} videos with total_frames={total_frames}")
    for v, c, t in zip(vids, counts, target):
        print(f"  {v.name}: frames={c}, samples={t}")

    # Extract images
    global_idx = 0
    total_saved = 0
    for v, c, t in zip(vids, counts, target):
        if t <= 0 or c <= 0:
            continue
        # choose frame indices evenly
        if t >= c:
            indices = list(range(c))
            # if too many, sample with replacement
            if t > c:
                indices = [random.randrange(0, c) for _ in range(t)]
        else:
            indices = [int(round(x)) for x in numpy_linspace(0, c - 1, t)]

        saved = extract_frames_from_video(v, out_dir, indices, args.imgsz, start_idx=global_idx)
        global_idx += saved
        total_saved += saved
        print(f"Saved {saved} images from {v.name} (global total={total_saved})")

    print(f"Done. Total images saved: {total_saved} in {out_dir}")


def numpy_linspace(a, b, n):
    # small helper to avoid numpy dependency for simple linspace
    if n <= 1:
        return [int(round(a))]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


if __name__ == "__main__":
    main()
