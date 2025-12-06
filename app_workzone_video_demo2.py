import time
from pathlib import Path
import tempfile
from collections import Counter
from typing import Optional, Tuple, List

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

DEFAULT_WEIGHTS_PATH = "bestv12.pt"   # change if needed
DEMO_VIDEOS_DIR = Path("data/demo")


def list_demo_videos() -> List[Path]:
    if not DEMO_VIDEOS_DIR.exists():
        return []
    return sorted(DEMO_VIDEOS_DIR.glob("*.mp4"))


# Semantic groups based on model.names (49 classes)
CHANNELIZATION = {
    "Cone",
    "Drum",
    "Barricade",
    "Barrier",
    "Vertical Panel",
    "Tubular Marker",
    "Fence",
}

WORKERS = {
    "Worker",
    "Police Officer",
}

VEHICLES = {
    "Work Vehicle",
    "Police Vehicle",
}

# Means/stds from Notebook 6 (val statistics)
MEAN_STD = {
    "frac_channelization": (0.517381, 0.336453),
    "frac_workers": (0.344524, 0.330260),
    "frac_vehicles": (0.109019, 0.241222),
}

WEIGHTS = {
    "frac_channelization": 0.9,
    "frac_workers": 0.7,
    "frac_vehicles": 0.4,
}

SMOOTH_ALPHA = 0.3  # for EMA smoothing over frames


# =========================
# HELPERS: model loading
# =========================

@st.cache_resource
def load_model_cached(weights_bytes: bytes, suffix: str, device: str):
    tmp_dir = Path(tempfile.gettempdir())
    tmp_path = tmp_dir / f"uploaded_yolo_weights{suffix}"
    with open(tmp_path, "wb") as f:
        f.write(weights_bytes)
    model = YOLO(str(tmp_path))
    try:
        model.to(device)
    except Exception:
        pass
    return model


@st.cache_resource
def load_model_default(weights_path: str, device: str):
    model = YOLO(weights_path)
    try:
        model.to(device)
    except Exception:
        pass
    return model


def resolve_device(device_choice: str) -> str:
    if device_choice == "GPU (cuda)":
        return "cuda"
    if device_choice == "CPU":
        return "cpu"

    # Auto
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# =========================
# HELPERS: scoring and drawing
# =========================

def logistic(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def frame_features_and_score(class_names: List[str]) -> Tuple[dict, float, float]:
    """
    Input: list of YOLO class names for this frame.
    Returns: features_dict, raw_score, score (0-1)
    """
    total = len(class_names)
    group_counts = Counter()

    for name in class_names:
        if name in CHANNELIZATION:
            group_counts["channelization"] += 1
        elif name in WORKERS:
            group_counts["workers"] += 1
        elif name in VEHICLES:
            group_counts["vehicles"] += 1
        else:
            group_counts["other"] += 1

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

    raw = 0.0
    for name, value in vals.items():
        mean, std = MEAN_STD[name]
        z = (value - mean) / (std + 1e-6)
        raw += WEIGHTS[name] * float(z)

    score = float(logistic(raw))

    features = {
        "total_objs": total,
        "count_channelization": group_counts["channelization"],
        "count_workers": group_counts["workers"],
        "count_vehicles": group_counts["vehicles"],
        "frac_channelization": frac_channelization,
        "frac_workers": frac_workers,
        "frac_vehicles": frac_vehicles,
    }

    return features, raw, score


def draw_workzone_banner(frame: np.ndarray, score: float) -> np.ndarray:
    h, w = frame.shape[:2]
    banner_h = int(0.12 * h)

    if score >= 0.65:
        label = f"WORK ZONE - HIGH RISK (score={score:.2f})"
        color = (0, 0, 255)   # red
    elif score >= 0.45:
        label = f"WORK ZONE (score={score:.2f})"
        color = (0, 165, 255)  # orange
    else:
        label = f"NO WORK ZONE (score={score:.2f})"
        color = (0, 128, 0)   # green

    cv2.rectangle(frame, (0, 0), (w, banner_h), color, thickness=-1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, scale, thickness)
    text_x = max(10, int((w - text_size[0]) / 2))
    text_y = int(banner_h * 0.7)

    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return frame


def plot_scores(raw_scores: List[float], smooth_scores: List[float]):
    fig, ax = plt.subplots(figsize=(7, 3))
    frames = np.arange(len(raw_scores))
    ax.plot(frames, raw_scores, label="raw", linewidth=1)
    ax.plot(frames, smooth_scores, label="smoothed", linewidth=2)
    ax.set_xlabel("frame index")
    ax.set_ylabel("workzone_score [0,1]")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.set_title("Work zone score over frames")
    st.pyplot(fig)


# =========================
# VIDEO PROCESSING
# =========================

def process_video_batch(
    input_path: Path,
    model: YOLO,
    conf: float,
    iou: float,
    device: str,
    max_frames: Optional[int] = None,
):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Could not open video.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_limit = min(total_frames, max_frames) if max_frames else total_frames

    tmp_dir = Path(tempfile.gettempdir())
    output_path = tmp_dir / "workzone_annotated_v2.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    progress = st.progress(0)
    st.write(
        f"Video info: {width}x{height} at {fps:.1f} fps, "
        f"{total_frames} frames, processing {frame_limit} frames."
    )

    frame_idx = 0
    raw_scores = []
    smooth_scores = []
    smooth_prev = None

    global_counts = Counter()

    while True:
        if frame_idx >= frame_limit:
            break
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=conf,
            iou=iou,
            verbose=False,
            device=device,
        )
        r = results[0]
        annotated = r.plot()

        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.int().cpu().tolist()
            names = [model.names[int(cid)] for cid in cls_ids]
        else:
            cls_ids = []
            names = []

        features, raw, score = frame_features_and_score(names)
        raw_scores.append(raw)

        if smooth_prev is None:
            smooth = score
        else:
            smooth = SMOOTH_ALPHA * score + (1.0 - SMOOTH_ALPHA) * smooth_prev
        smooth_scores.append(smooth)
        smooth_prev = smooth

        # Update global counts (by group)
        global_counts["frames"] += 1
        global_counts["total_objs"] += features["total_objs"]
        global_counts["channelization"] += features["count_channelization"]
        global_counts["workers"] += features["count_workers"]
        global_counts["vehicles"] += features["count_vehicles"]

        annotated = draw_workzone_banner(annotated, smooth)

        out.write(annotated)

        frame_idx += 1
        if frame_idx % 5 == 0 or frame_idx == frame_limit:
            progress.progress(min(frame_idx / frame_limit, 1.0))

    cap.release()
    out.release()
    progress.progress(1.0)

    if raw_scores:
        global_score = float(np.mean(smooth_scores))
    else:
        global_score = 0.0

    return output_path, global_score, fps, global_counts, raw_scores, smooth_scores


def run_live_preview(
    input_path: Path,
    model: YOLO,
    conf: float,
    iou: float,
    device: str,
    max_frames: Optional[int] = None,
):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_limit = min(total_frames, max_frames) if max_frames else total_frames

    st.write(
        f"Video info: {width}x{height} at {fps:.1f} fps, "
        f"{total_frames} frames, live preview of {frame_limit} frames."
    )

    frame_placeholder = st.empty()
    info_placeholder = st.empty()
    progress = st.progress(0)

    raw_scores = []
    smooth_scores = []
    smooth_prev = None

    frame_idx = 0
    delay = 1.0 / fps if fps > 0 else 0.03

    while True:
        if frame_idx >= frame_limit:
            break
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=conf,
            iou=iou,
            verbose=False,
            device=device,
        )
        r = results[0]
        annotated = r.plot()

        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.int().cpu().tolist()
            names = [model.names[int(cid)] for cid in cls_ids]
        else:
            names = []

        _, raw, score = frame_features_and_score(names)
        raw_scores.append(raw)

        if smooth_prev is None:
            smooth = score
        else:
            smooth = SMOOTH_ALPHA * score + (1.0 - SMOOTH_ALPHA) * smooth_prev
        smooth_scores.append(smooth)
        smooth_prev = smooth

        annotated = draw_workzone_banner(annotated, smooth)
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        info_placeholder.markdown(
            f"Frame {frame_idx + 1}/{frame_limit} | "
            f"Current smoothed work zone score: **{smooth:.2f}**"
        )

        frame_idx += 1
        if frame_idx % 5 == 0 or frame_idx == frame_limit:
            progress.progress(min(frame_idx / frame_limit, 1.0))

        time.sleep(delay)

    cap.release()
    progress.progress(1.0)

    if smooth_scores:
        final_score = float(np.mean(smooth_scores))
    else:
        final_score = 0.0

    st.success(f"Finished live preview. Final work zone score: {final_score:.2f}")


# =========================
# STREAMLIT UI
# =========================

def main():
    st.set_page_config(page_title="Work Zone Detector v2", layout="wide")
    st.title("Work Zone Detection - YOLOv12 + semantic score")

    st.markdown(
        "This app runs YOLOv12 on a video, groups detections into "
        "channelization, workers and vehicles, and computes a continuous "
        "workzone_score using z scored fractions plus a logistic function. "
        "Scores are smoothed over time to reduce flicker."
    )

    # Sidebar: device and model
    st.sidebar.header("Model settings")

    device_choice = st.sidebar.radio(
        "Device",
        ["Auto", "GPU (cuda)", "CPU"],
        index=0,
    )
    device = resolve_device(device_choice)
    st.sidebar.text(f"Using device: {device}")

    use_uploaded_weights = st.sidebar.checkbox(
        "Upload YOLO weights (.pt)",
        value=False,
    )

    uploaded_weights = None
    if use_uploaded_weights:
        uploaded_weights = st.sidebar.file_uploader(
            "Upload YOLO weights file",
            type=["pt"],
        )
    else:
        st.sidebar.text("Using default weights path:")
        st.sidebar.code(DEFAULT_WEIGHTS_PATH, language="text")

    conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
    iou = st.sidebar.slider("IoU threshold", 0.1, 0.9, 0.5, 0.05)
    max_frames = st.sidebar.number_input(
        "Max frames to process (0 for all)",
        min_value=0,
        value=0,
        step=50,
    )

    mode = st.sidebar.radio(
        "Run mode",
        ["Batch (save annotated video)", "Live preview"],
        index=0,
    )

    st.subheader("Video source")
    video_source = st.radio(
        "Choose video input",
        ["Upload video", "Use demo video"],
        index=0,
    )

    demo_videos = list_demo_videos()
    selected_demo = None
    video_file = None

    if video_source == "Upload video":
        video_file = st.file_uploader(
            "Video file",
            type=["mp4", "mov", "avi", "mkv"],
        )
    else:
        if not demo_videos:
            st.warning("No demo videos found in data/demo.")
        else:
            demo_names = [p.name for p in demo_videos]
            selected_name = st.selectbox("Select a demo video", demo_names)
            if selected_name:
                selected_demo = DEMO_VIDEOS_DIR / selected_name

    run_clicked = st.button("Run work zone detection", type="primary")

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
    with st.spinner("Loading YOLOv12 model..."):
        try:
            if use_uploaded_weights:
                if uploaded_weights is None:
                    st.error("Please upload YOLO weights in the sidebar.")
                    return
                model = load_model_cached(
                    uploaded_weights.read(),
                    suffix=Path(uploaded_weights.name).suffix,
                    device=device,
                )
            else:
                if not Path(DEFAULT_WEIGHTS_PATH).exists():
                    st.error("Default weights path does not exist.")
                    return
                model = load_model_default(DEFAULT_WEIGHTS_PATH, device=device)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

    st.success(f"Model loaded on device: {device}")

    frames_limit = None if max_frames == 0 else int(max_frames)

    if mode == "Batch (save annotated video)":
        st.info("Processing video in batch mode.")
        result = process_video_batch(
            input_path=tmp_video_path,
            model=model,
            conf=conf,
            iou=iou,
            device=device,
            max_frames=frames_limit,
        )
        if result is None:
            return
        (
            output_path,
            global_score,
            fps,
            counts,
            raw_scores,
            smooth_scores,
        ) = result

        st.subheader("Results")
        st.write(f"Processed frames: {counts['frames']} at {fps:.1f} fps")
        st.write(f"Global work zone score (smoothed mean): **{global_score:.2f}**")

        avg_objs = counts["total_objs"] / max(counts["frames"], 1)
        st.write(f"Average objects per frame: {avg_objs:.1f}")
        st.write(
            f"Total channelization: {counts['channelization']}, "
            f"workers: {counts['workers']}, "
            f"vehicles: {counts['vehicles']}"
        )

        st.write("Score curve:")
        plot_scores(raw_scores, smooth_scores)

        st.write("Annotated video:")
        st.video(str(output_path))
        with open(output_path, "rb") as f:
            st.download_button(
                label="Download annotated video",
                data=f,
                file_name="workzone_annotated_v2.mp4",
                mime="video/mp4",
            )
    else:
        st.info("Running live preview mode.")
        run_live_preview(
            input_path=tmp_video_path,
            model=model,
            conf=conf,
            iou=iou,
            device=device,
            max_frames=frames_limit,
        )


if __name__ == "__main__":
    main()