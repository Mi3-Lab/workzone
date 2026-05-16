import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from .data_models import StateIntervals, VideoGroundTruth, VideoPredictions


def _normalize_intervals(intervals: List[List[int]]) -> List[Tuple[int, int]]:
    cleaned: List[Tuple[int, int]] = []
    for start, end in intervals:
        if start > end:
            start, end = end, start
        cleaned.append((int(start), int(end)))
    return sorted(cleaned)


def load_ground_truth(path: str) -> Dict[str, VideoGroundTruth]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Ground-truth JSON must be an object keyed by video filename.")
    gt: Dict[str, VideoGroundTruth] = {}
    for video, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Ground-truth entry for {video} must be an object.")
        states: StateIntervals = {}
        for state, intervals in entry.items():
            if intervals is None:
                continue
            if not isinstance(intervals, list):
                raise ValueError(f"Ground-truth intervals for {video}:{state} must be a list.")
            states[state] = _normalize_intervals(intervals)
        gt[video] = VideoGroundTruth(states=states)
    return gt


def load_predictions(path: str) -> Dict[str, VideoPredictions]:
    path_obj = Path(path)
    if path_obj.is_dir():
        return load_predictions_from_timeline_dir(path)
    if path_obj.suffix.lower() == ".csv":
        return load_predictions_from_timeline_csv(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Predictions JSON must be an object keyed by video filename.")
    preds: Dict[str, VideoPredictions] = {}
    for video, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Prediction entry for {video} must be an object.")
        fps = entry.get("fps")
        states_raw = entry.get("states")
        states: Optional[StateIntervals] = None
        if isinstance(states_raw, dict):
            states = {k: _normalize_intervals(v) for k, v in states_raw.items()}
        preds[video] = VideoPredictions(
            states=states,
            fps=float(fps) if fps is not None else None,
            detections=entry.get("detections"),
            ocr=entry.get("ocr"),
        )
    return preds


def load_predictions_from_timeline_dir(path: str) -> Dict[str, VideoPredictions]:
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"Timeline directory not found: {path}")
    preds: Dict[str, VideoPredictions] = {}
    csv_paths = list(root.rglob("*_timeline*.csv"))
    if not csv_paths:
        csv_paths = list(root.rglob("*.csv"))
    for csv_path in sorted(csv_paths):
        preds.update(load_predictions_from_timeline_csv(str(csv_path)))
    if not preds:
        raise ValueError(f"No timeline CSVs found under: {path}")
    return preds


def load_predictions_from_timeline_csv(path: str) -> Dict[str, VideoPredictions]:
    path_obj = Path(path)
    rows: List[Tuple[int, str, Optional[float]]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            lower = {k.lower(): v for k, v in row.items() if k is not None}
            if "frame" not in lower or "state" not in lower:
                continue
            frame = int(float(lower["frame"]))
            state = _normalize_state_label(lower["state"])
            time_sec = lower.get("time_sec")
            time_val = float(time_sec) if time_sec not in (None, "") else None
            rows.append((frame, state, time_val))
    if not rows:
        raise ValueError(f"No valid rows with frame/state found in {path}")

    rows.sort(key=lambda r: r[0])
    frames = [r[0] for r in rows]
    states = [r[1] for r in rows]
    times = [r[2] for r in rows]

    max_frame = frames[-1]
    labels = ["outside"] * (max_frame + 1)

    current_label = states[0]
    last_frame = frames[0]
    labels[: last_frame + 1] = [current_label] * (last_frame + 1)
    for idx, frame in enumerate(frames):
        label = states[idx]
        if frame > last_frame:
            labels[last_frame:frame + 1] = [current_label] * (frame - last_frame + 1)
        labels[frame] = label
        current_label = label
        last_frame = frame

    fps = _estimate_fps(frames, times)
    intervals = _intervals_from_labels(labels)

    video_name = path_obj.stem
    # Batch SOTA runner prefixes timeline files with "sota_".
    if video_name.startswith("sota_"):
        video_name = video_name[len("sota_") :]
    for suffix in ("_timeline_fusion", "_timeline", "_calibrated"):
        if video_name.endswith(suffix):
            video_name = video_name[: -len(suffix)]
            break
    if not video_name.endswith(".mp4"):
        video_name = f"{video_name}.mp4"

    return {
        video_name: VideoPredictions(
            states=intervals,
            fps=fps,
            detections=None,
            ocr=None,
        )
    }


def _normalize_state_label(label: str) -> str:
    if label is None:
        return "outside"
    value = str(label).strip().lower()
    mapping = {
        "out": "outside",
        "outside": "outside",
        "approaching": "approaching",
        "approach": "approaching",
        "inside": "inside",
        "in": "inside",
        "exiting": "exiting",
        "exit": "exiting",
    }
    return mapping.get(value, value)


def _intervals_from_labels(labels: List[str]) -> StateIntervals:
    intervals: StateIntervals = {}
    if not labels:
        return intervals
    start = 0
    current = labels[0]
    for idx in range(1, len(labels)):
        if labels[idx] != current:
            intervals.setdefault(current, []).append((start, idx - 1))
            start = idx
            current = labels[idx]
    intervals.setdefault(current, []).append((start, len(labels) - 1))
    return intervals


def _estimate_fps(frames: List[int], times: List[Optional[float]]) -> Optional[float]:
    samples: List[float] = []
    for i in range(1, len(frames)):
        t0 = times[i - 1]
        t1 = times[i]
        if t0 is None or t1 is None:
            continue
        dt = t1 - t0
        df = frames[i] - frames[i - 1]
        if dt > 0 and df > 0:
            samples.append(df / dt)
    if not samples:
        return None
    return statistics.median(samples)
