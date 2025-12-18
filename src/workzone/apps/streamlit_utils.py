"""Shared utilities for Streamlit applications."""

import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from src.workzone.utils.logging_config import setup_logger

logger = setup_logger(__name__)


# ============================================================
# WORKZONE CLASS DEFINITIONS
# ============================================================

WORKZONE_CLASSES_10 = {
    0: "Cone",
    1: "Drum",
    2: "Barricade",
    3: "Barrier",
    4: "Vertical Panel",
    5: "Work Vehicle",
    6: "Worker",
    7: "Arrow Board",
    8: "Temporary Traffic Control Message Board",
    9: "Temporary Traffic Control Sign",
}

# Semantic groupings for 50-class model
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

MESSAGE_BOARD = {
    "Temporary Traffic Control Message Board",
    "Arrow Board",
}

OTHER_ROADWORK = {
    "Work Equipment",
    "Other Roadwork Objects",
}


def is_ttc_sign(name: str) -> bool:
    """Check if class name is a Temporary Traffic Control Sign."""
    return name.startswith("Temporary Traffic Control Sign")


# ============================================================
# MATH UTILITIES
# ============================================================

def ema(prev: float, x: float, alpha: float) -> float:
    """
    Exponential moving average.

    Args:
        prev: Previous EMA value (None for first value)
        x: Current value
        alpha: Smoothing factor (0-1)

    Returns:
        Updated EMA value
    """
    if prev is None:
        return x
    return alpha * x + (1.0 - alpha) * prev


def clamp01(x: float) -> float:
    """Clamp value to [0, 1]."""
    return float(max(0.0, min(1.0, x)))


def safe_div(a: float, b: float) -> float:
    """Safe division (returns 0 if denominator is 0)."""
    return float(a / b) if b > 0 else 0.0


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model_cached(weights_bytes: bytes, suffix: str, device: str) -> YOLO:
    """
    Load and cache YOLO model from uploaded weights.

    Args:
        weights_bytes: Bytes of model weights
        suffix: File suffix for temp file
        device: Device to load model on

    Returns:
        Loaded YOLO model
    """
    tmp_dir = Path(tempfile.gettempdir())
    tmp_path = tmp_dir / f"uploaded_yolo_weights{suffix}"

    with open(tmp_path, "wb") as f:
        f.write(weights_bytes)

    logger.info(f"Loading model from temporary file: {tmp_path}")
    model = YOLO(str(tmp_path))

    try:
        model.to(device)
        logger.info(f"Model moved to {device}")
    except Exception as e:
        logger.warning(f"Could not move model to {device}: {e}")

    return model


@st.cache_resource
def load_model_default(weights_path: str, device: str) -> YOLO:
    """
    Load and cache YOLO model from default path.

    Args:
        weights_path: Path to model weights
        device: Device to load model on

    Returns:
        Loaded YOLO model
    """
    logger.info(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)

    try:
        model.to(device)
        logger.info(f"Model moved to {device}")
    except Exception as e:
        logger.warning(f"Could not move model to {device}: {e}")

    return model


# ============================================================
# DEVICE UTILITIES
# ============================================================

def resolve_device(device_choice: str) -> str:
    """
    Resolve device choice to proper torch device string.

    Args:
        device_choice: Human-readable device choice

    Returns:
        PyTorch device string
    """
    import torch

    if device_choice == "CPU":
        return "cpu"
    elif device_choice == "CUDA (GPU)":
        if torch.cuda.is_available():
            return "cuda"
        else:
            logger.warning("CUDA not available, falling back to CPU")
            st.warning("CUDA not available, using CPU instead")
            return "cpu"
    elif device_choice.startswith("CUDA:"):
        # Extract GPU number
        try:
            gpu_id = device_choice.split(":")[1]
            if torch.cuda.is_available():
                return f"cuda:{gpu_id}"
            else:
                logger.warning("CUDA not available, falling back to CPU")
                st.warning("CUDA not available, using CPU instead")
                return "cpu"
        except Exception:
            return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        return "cpu"


# ============================================================
# VIDEO UTILITIES
# ============================================================

def list_demo_videos(demo_dir: Path) -> List[Path]:
    """
    List available demo videos in a directory.

    Args:
        demo_dir: Directory containing demo videos

    Returns:
        List of video file paths
    """
    if not demo_dir.exists():
        logger.warning(f"Demo directory not found: {demo_dir}")
        return []

    videos = sorted(demo_dir.glob("*.mp4"))
    logger.info(f"Found {len(videos)} demo videos in {demo_dir}")
    return videos


def get_video_properties(video_path: Path) -> Tuple[int, int, float, int]:
    """
    Get video properties.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height, fps, frame_count)
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return width, height, fps, frame_count


# ============================================================
# DRAWING UTILITIES
# ============================================================

def draw_detection_boxes(
    frame: np.ndarray,
    boxes: np.ndarray,
    class_names: List[str],
    confidences: np.ndarray,
) -> np.ndarray:
    """
    Draw bounding boxes on frame.

    Args:
        frame: Input frame
        boxes: Bounding boxes (N, 4)
        class_names: Class names (N,)
        confidences: Confidence scores (N,)

    Returns:
        Frame with drawn boxes
    """
    for box, class_name, confidence in zip(boxes, class_names, confidences):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return frame


def draw_workzone_banner(
    frame: np.ndarray,
    score: float,
    status_text: str = "Work Zone Score",
) -> np.ndarray:
    """
    Draw work zone score banner on frame.

    Args:
        frame: Input frame
        score: Work zone score (0-1)
        status_text: Status text to display

    Returns:
        Frame with banner
    """
    height, width = frame.shape[:2]

    # Create semi-transparent overlay
    overlay = frame.copy()
    banner_height = 120
    cv2.rectangle(overlay, (0, 0), (width, banner_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw text
    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    # Score with color coding
    color = (
        (0, 255, 0) if score > 0.7 else (0, 255, 255) if score > 0.4 else (0, 0, 255)
    )
    cv2.putText(
        frame,
        f"Score: {score:.3f}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3,
    )

    return frame


# ============================================================
# SCORING UTILITIES
# ============================================================

def logistic(x: float) -> float:
    """Logistic sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def compute_simple_workzone_score(class_ids: np.ndarray) -> float:
    """
    Compute simple work zone score (0-1) based on class presence.

    Args:
        class_ids: Array of detected class IDs

    Returns:
        Work zone score between 0 and 1
    """
    if class_ids.size == 0:
        return 0.0

    # Weight classes by importance
    weights = {
        0: 1.0,  # Cone
        1: 1.0,  # Drum
        2: 1.2,  # Barricade
        3: 1.2,  # Barrier
        4: 0.8,  # Vertical Panel
        5: 1.5,  # Work Vehicle
        6: 1.8,  # Worker
        7: 1.0,  # Arrow Board
        8: 1.3,  # Message Board
        9: 0.9,  # Sign
    }

    total_weight = sum(weights.get(cid, 0.5) for cid in class_ids)
    normalized_score = min(1.0, total_weight / 10.0)

    return normalized_score
