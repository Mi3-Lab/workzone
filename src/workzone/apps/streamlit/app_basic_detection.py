"""
Streamlit application for basic YOLO work zone detection.

This app provides a simple interface for running YOLO detection on videos
with work zone scoring and class counting.
"""

import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO

from workzone.apps.streamlit_utils import (
    WORKZONE_CLASSES_10,
    compute_simple_workzone_score,
    draw_workzone_banner,
    list_demo_videos,
    load_model_cached,
    load_model_default,
    resolve_device,
)
from workzone.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Configuration
DEFAULT_WEIGHTS_PATH = "weights/best.pt"
DEMO_VIDEOS_DIR = Path("data/demo")


def process_video_batch(
    input_path: Path,
    model: YOLO,
    conf: float,
    iou: float,
    device: str,
    max_frames: Optional[int] = None,
) -> Tuple[Optional[Path], Optional[float], Optional[int], Optional[float], Dict]:
    """
    Process video in batch mode and save annotated output.

    Args:
        input_path: Path to input video
        model: Loaded YOLO model
        conf: Confidence threshold
        iou: IOU threshold
        device: Device to run on
        max_frames: Maximum frames to process (None for all)

    Returns:
        Tuple of (output_path, global_score, frame_count, fps, class_counts)
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Could not open video.")
        return None, None, None, None, {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_limit = min(total_frames, max_frames) if max_frames else total_frames

    # Setup output video
    tmp_dir = Path(tempfile.gettempdir())
    output_path = tmp_dir / "workzone_annotated_batch.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    progress = st.progress(0)
    st.write(
        f"Video info: {width}x{height} at {fps:.1f} fps, "
        f"{total_frames} total frames, processing {frame_limit} frames."
    )

    frame_idx = 0
    workzone_scores = []
    class_counts = {cid: 0 for cid in WORKZONE_CLASSES_10.keys()}

    # Process frames
    while frame_idx < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model.predict(frame, conf=conf, iou=iou, verbose=False, device=device)
        result = results[0]
        annotated_frame = result.plot()

        # Compute score and count classes
        if result.boxes is not None and len(result.boxes) > 0:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            score = compute_simple_workzone_score(cls_ids)
            workzone_scores.append(score)

            for cid in cls_ids:
                if cid in class_counts:
                    class_counts[cid] += 1
        else:
            score = 0.0
            workzone_scores.append(0.0)

        # Draw banner
        annotated_frame = draw_workzone_banner(annotated_frame, score)
        out.write(annotated_frame)

        frame_idx += 1
        if frame_idx % 5 == 0:
            progress.progress(min(frame_idx / frame_limit, 1.0))

    cap.release()
    out.release()
    progress.progress(1.0)

    global_score = float(np.mean(workzone_scores)) if workzone_scores else 0.0

    logger.info(f"Processed {frame_idx} frames, global score: {global_score:.2f}")

    return output_path, global_score, frame_idx, fps, class_counts


def run_live_preview(
    input_path: Path,
    model: YOLO,
    conf: float,
    iou: float,
    device: str,
    max_frames: Optional[int] = None,
) -> None:
    """
    Process video in live preview mode (no file saved).

    Args:
        input_path: Path to input video
        model: Loaded YOLO model
        conf: Confidence threshold
        iou: IOU threshold
        device: Device to run on
        max_frames: Maximum frames to process (None for all)
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_limit = min(total_frames, max_frames) if max_frames else total_frames

    st.write(f"Live preview of {frame_limit} frames at {fps:.1f} fps")

    frame_placeholder = st.empty()
    info_placeholder = st.empty()
    progress = st.progress(0)

    workzone_scores = []
    class_counts = {cid: 0 for cid in WORKZONE_CLASSES_10.keys()}
    frame_idx = 0
    delay = 1.0 / fps if fps > 0 else 0.03

    while frame_idx < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model.predict(frame, conf=conf, iou=iou, verbose=False, device=device)
        result = results[0]
        annotated_frame = result.plot()

        # Compute score
        if result.boxes is not None and len(result.boxes) > 0:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            score = compute_simple_workzone_score(cls_ids)
            workzone_scores.append(score)

            for cid in cls_ids:
                if cid in class_counts:
                    class_counts[cid] += 1
        else:
            score = 0.0
            workzone_scores.append(0.0)

        annotated_frame = draw_workzone_banner(annotated_frame, score)

        # Display frame
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        current_score = float(np.mean(workzone_scores)) if workzone_scores else 0.0
        info_placeholder.markdown(
            f"Frame {frame_idx + 1}/{frame_limit} | "
            f"Current work zone score: **{current_score:.2f}**"
        )

        frame_idx += 1
        if frame_idx % 5 == 0:
            progress.progress(min(frame_idx / frame_limit, 1.0))

        time.sleep(delay)

    cap.release()
    progress.progress(1.0)

    final_score = float(np.mean(workzone_scores)) if workzone_scores else 0.0
    st.success(f"Finished live preview. Final work zone score: {final_score:.2f}")

    # Display class counts
    st.write("Class counts:")
    for cid, name in WORKZONE_CLASSES_10.items():
        count = class_counts.get(cid, 0)
        if count > 0:
            st.text(f"{name}: {count}")


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(page_title="Work Zone YOLO Video Demo", layout="wide")
    st.title("Work Zone Detection - YOLO Video Demo")

    st.markdown(
        """
        Upload a video or choose a demo video, select YOLO weights, device, and thresholds.
        Run detection in batch mode (saves annotated video) or live preview mode.
        """
    )

    # Sidebar - Model settings
    st.sidebar.header("Model Settings")

    device_choice = st.sidebar.radio(
        "Device", ["CPU", "CUDA (GPU)"], index=1 if torch.cuda.is_available() else 0
    )
    device = resolve_device(device_choice)
    st.sidebar.text(f"Using: {device}")

    use_uploaded_weights = st.sidebar.checkbox("Upload YOLO weights (.pt)", value=False)

    if use_uploaded_weights:
        uploaded_weights = st.sidebar.file_uploader("Upload YOLO weights", type=["pt"])
    else:
        st.sidebar.text("Default weights:")
        st.sidebar.code(DEFAULT_WEIGHTS_PATH)
        uploaded_weights = None

    conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
    iou = st.sidebar.slider("IoU threshold", 0.1, 0.9, 0.5, 0.05)
    max_frames = st.sidebar.number_input(
        "Max frames (0 for all)", min_value=0, value=0, step=50
    )

    mode = st.sidebar.radio(
        "Mode", ["Batch (save video)", "Live preview"], index=0
    )

    # Main - Video source
    st.subheader("Video Source")

    video_source = st.radio("Choose input", ["Upload video", "Use demo video"], index=0)

    selected_demo = None
    video_file = None

    if video_source == "Upload video":
        video_file = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
    else:
        demo_videos = list_demo_videos(DEMO_VIDEOS_DIR)
        if not demo_videos:
            st.warning("No demo videos found in data/demo")
        else:
            demo_names = [p.name for p in demo_videos]
            selected_name = st.selectbox("Select demo video", demo_names)
            if selected_name:
                selected_demo = DEMO_VIDEOS_DIR / selected_name

    run_clicked = st.button("Run Detection", type="primary")

    if run_clicked:
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
        with st.spinner("Loading YOLO model..."):
            try:
                if use_uploaded_weights:
                    if uploaded_weights is None:
                        st.error("Please upload weights in sidebar.")
                        return
                    model = load_model_cached(
                        uploaded_weights.read(),
                        Path(uploaded_weights.name).suffix,
                        device,
                    )
                else:
                    if not Path(DEFAULT_WEIGHTS_PATH).exists():
                        st.error(f"Weights not found: {DEFAULT_WEIGHTS_PATH}")
                        return
                    model = load_model_default(DEFAULT_WEIGHTS_PATH, device)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                logger.error(f"Model loading error: {e}")
                return

        st.success(f"Model loaded on {device}")

        frames_limit = None if max_frames == 0 else int(max_frames)

        # Process video
        if mode == "Batch (save video)":
            st.info("Processing video in batch mode...")
            output_path, global_score, processed_frames, fps, class_counts = (
                process_video_batch(tmp_video_path, model, conf, iou, device, frames_limit)
            )

            if output_path is None:
                return

            st.subheader("Results")
            st.write(f"Processed: {processed_frames} frames at {fps:.1f} fps")
            st.write(f"Global work zone score: **{global_score:.2f}**")

            st.write("Class counts:")
            for cid, name in WORKZONE_CLASSES_10.items():
                count = class_counts.get(cid, 0)
                if count > 0:
                    st.text(f"{name}: {count}")

            st.write("Annotated video:")
            st.video(str(output_path))

            with open(output_path, "rb") as f:
                st.download_button(
                    "Download Annotated Video",
                    data=f,
                    file_name="workzone_annotated.mp4",
                    mime="video/mp4",
                )
        else:
            st.info("Running live preview mode...")
            run_live_preview(tmp_video_path, model, conf, iou, device, frames_limit)


if __name__ == "__main__":
    main()
