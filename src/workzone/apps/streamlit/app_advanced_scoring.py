"""
Streamlit application for advanced work zone detection with semantic scoring.

This app provides semantic grouping of detections (channelization, workers, vehicles),
statistical normalization, weighted scoring, and EMA smoothing for temporal consistency.
"""

import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO

from src.workzone.apps.streamlit_utils import (
    CHANNELIZATION,
    VEHICLES,
    WORKERS,
    list_demo_videos,
    load_model_cached,
    load_model_default,
    logistic,
    resolve_device,
)
from src.workzone.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Configuration
DEFAULT_WEIGHTS_PATH = "weights/bestv12.pt"
DEMO_VIDEOS_DIR = Path("data/demo")

# Statistical parameters from validation dataset
MEAN_STD = {
    "frac_channelization": (0.517381, 0.336453),
    "frac_workers": (0.344524, 0.330260),
    "frac_vehicles": (0.109019, 0.241222),
}

# Feature weights for scoring
WEIGHTS = {
    "frac_channelization": 0.9,
    "frac_workers": 0.7,
    "frac_vehicles": 0.4,
}

# EMA smoothing parameter
SMOOTH_ALPHA = 0.3


def compute_frame_features_and_score(
    class_names: List[str],
) -> Tuple[Dict[str, float], float, float]:
    """
    Compute semantic features and work zone score for a frame.

    Args:
        class_names: List of detected class names

    Returns:
        Tuple of (features_dict, raw_score, normalized_score)
    """
    total = len(class_names)
    group_counts = Counter()

    # Count objects in semantic groups
    for name in class_names:
        if name in CHANNELIZATION:
            group_counts["channelization"] += 1
        elif name in WORKERS:
            group_counts["workers"] += 1
        elif name in VEHICLES:
            group_counts["vehicles"] += 1
        else:
            group_counts["other"] += 1

    # Compute fractions
    if total == 0:
        frac_channelization = 0.0
        frac_workers = 0.0
        frac_vehicles = 0.0
    else:
        frac_channelization = group_counts["channelization"] / total
        frac_workers = group_counts["workers"] / total
        frac_vehicles = group_counts["vehicles"] / total

    vals = {
        "frac_channelization": frac_channelization,
        "frac_workers": frac_workers,
        "frac_vehicles": frac_vehicles,
    }

    # Compute weighted z-score
    raw_score = 0.0
    for name, value in vals.items():
        mean, std = MEAN_STD[name]
        z = (value - mean) / (std + 1e-6)
        raw_score += WEIGHTS[name] * float(z)

    # Apply logistic transformation
    normalized_score = float(logistic(raw_score))

    # Build features dictionary
    features = {
        "total_objs": total,
        "count_channelization": group_counts["channelization"],
        "count_workers": group_counts["workers"],
        "count_vehicles": group_counts["vehicles"],
        "frac_channelization": frac_channelization,
        "frac_workers": frac_workers,
        "frac_vehicles": frac_vehicles,
    }

    return features, raw_score, normalized_score


def draw_workzone_banner(frame: np.ndarray, score: float) -> np.ndarray:
    """
    Draw colored work zone banner with score.

    Args:
        frame: Input frame
        score: Work zone score (0-1)

    Returns:
        Frame with banner
    """
    h, w = frame.shape[:2]
    banner_h = int(0.12 * h)

    # Determine status and color
    if score >= 0.65:
        label = f"WORK ZONE - HIGH RISK (score={score:.2f})"
        color = (0, 0, 255)  # Red
    elif score >= 0.45:
        label = f"WORK ZONE (score={score:.2f})"
        color = (0, 165, 255)  # Orange
    else:
        label = f"NO WORK ZONE (score={score:.2f})"
        color = (0, 128, 0)  # Green

    # Draw banner
    cv2.rectangle(frame, (0, 0), (w, banner_h), color, thickness=-1)

    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, scale, thickness)
    text_x = max(10, int((w - text_size[0]) / 2))
    text_y = int(banner_h * 0.7)

    cv2.putText(
        frame, label, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )

    return frame


def plot_scores(raw_scores: List[float], smooth_scores: List[float]) -> None:
    """Plot raw and smoothed scores over frames."""
    fig, ax = plt.subplots(figsize=(7, 3))
    frames = np.arange(len(raw_scores))
    ax.plot(frames, raw_scores, label="Raw", linewidth=1, alpha=0.7)
    ax.plot(frames, smooth_scores, label="Smoothed (EMA)", linewidth=2)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Work Zone Score [0,1]")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.set_title("Work Zone Score Over Time")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def process_video_batch(
    input_path: Path,
    model: YOLO,
    conf: float,
    iou: float,
    device: str,
    max_frames: Optional[int] = None,
) -> Optional[Tuple]:
    """
    Process video in batch mode with advanced scoring.

    Returns:
        Tuple of (output_path, global_score, fps, counts, raw_scores, smooth_scores)
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Could not open video.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_limit = min(total_frames, max_frames) if max_frames else total_frames

    # Setup output
    tmp_dir = Path(tempfile.gettempdir())
    output_path = tmp_dir / "workzone_annotated_advanced.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    progress = st.progress(0)
    st.write(f"Processing {frame_limit} frames at {fps:.1f} fps ({width}x{height})")

    frame_idx = 0
    raw_scores = []
    smooth_scores = []
    smooth_val = 0.0

    counts = {
        "frames": 0,
        "total_objs": 0,
        "channelization": 0,
        "workers": 0,
        "vehicles": 0,
    }

    while frame_idx < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model.predict(frame, conf=conf, iou=iou, verbose=False, device=device)
        result = results[0]
        annotated_frame = result.plot()

        # Extract class names
        class_names = []
        if result.boxes is not None and len(result.boxes) > 0:
            for cid in result.boxes.cls.cpu().numpy():
                class_names.append(result.names[int(cid)])

        # Compute features and score
        features, raw_score, norm_score = compute_frame_features_and_score(class_names)
        raw_scores.append(norm_score)

        # Apply EMA smoothing
        if frame_idx == 0:
            smooth_val = norm_score
        else:
            smooth_val = SMOOTH_ALPHA * norm_score + (1 - SMOOTH_ALPHA) * smooth_val
        smooth_scores.append(smooth_val)

        # Update counts
        counts["frames"] += 1
        counts["total_objs"] += features["total_objs"]
        counts["channelization"] += features["count_channelization"]
        counts["workers"] += features["count_workers"]
        counts["vehicles"] += features["count_vehicles"]

        # Draw banner
        annotated_frame = draw_workzone_banner(annotated_frame, smooth_val)
        out.write(annotated_frame)

        frame_idx += 1
        if frame_idx % 5 == 0:
            progress.progress(min(frame_idx / frame_limit, 1.0))

    cap.release()
    out.release()
    progress.progress(1.0)

    global_score = float(np.mean(smooth_scores)) if smooth_scores else 0.0

    logger.info(f"Processed {frame_idx} frames, global score: {global_score:.2f}")

    return output_path, global_score, fps, counts, raw_scores, smooth_scores


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(page_title="Work Zone Advanced Scoring", layout="wide")
    st.title("Work Zone Detection - Advanced Semantic Scoring")

    st.markdown(
        """
        This app uses **semantic grouping** and **statistical normalization** for work zone scoring:
        - Groups detections: Channelization, Workers, Vehicles
        - Computes z-scored fractions with weighted aggregation
        - Applies **EMA smoothing** to reduce temporal flicker
        """
    )

    # Sidebar
    st.sidebar.header("Model Settings")

    device_choice = st.sidebar.radio(
        "Device", ["CPU", "CUDA (GPU)"], index=1 if torch.cuda.is_available() else 0
    )
    device = resolve_device(device_choice)
    st.sidebar.text(f"Using: {device}")

    use_uploaded_weights = st.sidebar.checkbox("Upload YOLO weights", value=False)

    if use_uploaded_weights:
        uploaded_weights = st.sidebar.file_uploader("Upload weights (.pt)", type=["pt"])
    else:
        st.sidebar.text("Default weights:")
        st.sidebar.code(DEFAULT_WEIGHTS_PATH)
        uploaded_weights = None

    conf = st.sidebar.slider("Confidence", 0.1, 0.9, 0.4, 0.05)
    iou = st.sidebar.slider("IoU", 0.1, 0.9, 0.5, 0.05)
    max_frames = st.sidebar.number_input("Max frames (0=all)", 0, value=0, step=50)

    mode = st.sidebar.radio("Mode", ["Batch (save video)", "Live preview"], index=0)

    # Main area
    st.subheader("Video Source")
    video_source = st.radio("Choose input", ["Upload video", "Use demo video"], index=0)

    selected_demo = None
    video_file = None

    if video_source == "Upload video":
        video_file = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
    else:
        demo_videos = list_demo_videos(DEMO_VIDEOS_DIR)
        if not demo_videos:
            st.warning("No demo videos found")
        else:
            demo_names = [p.name for p in demo_videos]
            selected_name = st.selectbox("Select demo", demo_names)
            if selected_name:
                selected_demo = DEMO_VIDEOS_DIR / selected_name

    run_clicked = st.button("Run Detection", type="primary")

    if not run_clicked:
        return

    # Resolve video path
    if video_source == "Upload video":
        if video_file is None:
            st.warning("Please upload a video first.")
            return
        tmp_video_path = Path(tempfile.gettempdir()) / video_file.name
        with open(tmp_video_path, "wb") as f:
            f.write(video_file.getbuffer())
    else:
        if selected_demo is None:
            st.warning("Please select a demo video.")
            return
        tmp_video_path = selected_demo

    # Load model
    with st.spinner("Loading model..."):
        try:
            if use_uploaded_weights:
                if uploaded_weights is None:
                    st.error("Please upload weights")
                    return
                model = load_model_cached(
                    uploaded_weights.read(), Path(uploaded_weights.name).suffix, device
                )
            else:
                if not Path(DEFAULT_WEIGHTS_PATH).exists():
                    st.error(f"Weights not found: {DEFAULT_WEIGHTS_PATH}")
                    return
                model = load_model_default(DEFAULT_WEIGHTS_PATH, device)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

    st.success(f"Model loaded on {device}")

    frames_limit = None if max_frames == 0 else int(max_frames)

    # Process
    if mode == "Batch (save video)":
        st.info("Processing in batch mode...")
        result = process_video_batch(
            tmp_video_path, model, conf, iou, device, frames_limit
        )

        if result is None:
            return

        output_path, global_score, fps, counts, raw_scores, smooth_scores = result

        st.subheader("Results")
        st.write(f"**Processed**: {counts['frames']} frames at {fps:.1f} fps")
        st.write(f"**Global Score** (smoothed mean): **{global_score:.2f}**")

        avg_objs = counts["total_objs"] / max(counts["frames"], 1)
        st.write(f"**Average objects/frame**: {avg_objs:.1f}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Channelization", counts["channelization"])
        with col2:
            st.metric("Workers", counts["workers"])
        with col3:
            st.metric("Vehicles", counts["vehicles"])

        st.write("**Score Timeline**:")
        plot_scores(raw_scores, smooth_scores)

        st.write("**Annotated Video**:")
        st.video(str(output_path))

        with open(output_path, "rb") as f:
            st.download_button(
                "Download Annotated Video",
                data=f,
                file_name="workzone_advanced.mp4",
                mime="video/mp4",
            )


if __name__ == "__main__":
    main()
