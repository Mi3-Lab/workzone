"""
Streamlit application for advanced work zone detection with CLIP fusion and state machine.

This app provides:
- YOLO timeline with semantic scoring
- CLIP-based semantic verification (triggered)
- Anti-flicker state machine (OUT → APPROACHING → INSIDE → EXITING)
- Live preview and batch processing modes
"""

import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO

from workzone.apps.streamlit_utils import (
    CHANNELIZATION,
    MESSAGE_BOARD,
    OTHER_ROADWORK,
    VEHICLES,
    WORKERS,
    clamp01,
    ema,
    is_ttc_sign,
    list_demo_videos,
    load_model_cached,
    load_model_default,
    logistic,
    resolve_device,
    safe_div,
)
from workzone.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Configuration
FUSION_BASELINE_WEIGHTS = "weights/yolo12s_fusion_baseline.pt"  # Fusion baseline (pre hard-neg)
HARDNEG_WEIGHTS = "weights/yolo12s_hardneg_1280.pt"  # Hard-negative trained @1280px
DEMO_VIDEOS_DIR = Path("data/demo")
VIDEOS_COMPRESSED_DIR = Path("data/videos_compressed")

# CLIP configuration
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"


# ------------------------------------------------------------
# Lightweight context cue + adaptive EMA helpers
# ------------------------------------------------------------
def orange_ratio_hsv(frame_bgr: np.ndarray, h_low: int = 5, h_high: int = 25, s_th: int = 80, v_th: int = 50) -> float:
    """Compute ratio of orange-like pixels in HSV space.
    Defaults roughly capture traffic-cone/barrier oranges.
    Returns [0,1]."""
    if frame_bgr is None or frame_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (h >= h_low) & (h <= h_high) & (s >= s_th) & (v >= v_th)
    ratio = float(mask.sum()) / float(mask.size)
    return clamp01(ratio)


def context_boost_from_orange(ratio: float, center: float = 0.08, k: float = 30.0) -> float:
    """Map orange ratio to a confidence-like score via logistic.
    "center" is the ratio where output ~0.5; "k" controls slope."""
    return clamp01(float(logistic(k * (ratio - center))))


def adaptive_alpha(evidence: float, alpha_min: float = 0.10, alpha_max: float = 0.50) -> float:
    """Interpolate EMA alpha based on evidence in [0,1]."""
    e = clamp01(float(evidence))
    return float(alpha_min + (alpha_max - alpha_min) * e)


@st.cache_resource
def load_clip_bundle(device: str) -> Tuple[bool, Optional[Dict]]:
    """Load OpenCLIP model and preprocessing."""
    try:
        import open_clip
        from PIL import Image

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

        model = model.to(device)
        model.eval()

        return True, {
            "open_clip": open_clip,
            "PIL_Image": Image,
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "model_name": CLIP_MODEL_NAME,
            "pretrained": CLIP_PRETRAINED,
        }
    except Exception as e:
        logger.warning(f"Failed to load CLIP: {e}")
        return False, None


def clip_text_embeddings(
    clip_bundle: Dict, device: str, pos_text: str, neg_text: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute CLIP text embeddings for positive and negative prompts."""
    open_clip = clip_bundle["open_clip"]
    model = clip_bundle["model"]
    tokenizer = clip_bundle["tokenizer"]

    texts = [pos_text, neg_text]
    tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        txt = model.encode_text(tokens)
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-8)

    return txt[0], txt[1]


def clip_frame_score(
    clip_bundle: Dict,
    device: str,
    frame_bgr: np.ndarray,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
) -> float:
    """
    Compute CLIP semantic score for frame.

    Returns difference between positive and negative similarity.
    """
    Image = clip_bundle["PIL_Image"]
    preprocess = clip_bundle["preprocess"]
    model = clip_bundle["model"]

    # BGR -> RGB -> PIL
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    x = preprocess(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        img = model.encode_image(x)
        img = img / (img.norm(dim=-1, keepdim=True) + 1e-8)

        pos = (img @ pos_emb.unsqueeze(-1)).squeeze().item()
        neg = (img @ neg_emb.unsqueeze(-1)).squeeze().item()

    return float(pos - neg)


def group_counts_from_names(names: List[str]) -> Dict[str, int]:
    """Count detections by semantic group."""
    c = Counter()
    for n in names:
        if n in CHANNELIZATION:
            c["channelization"] += 1
        elif n in WORKERS:
            c["workers"] += 1
        elif n in VEHICLES:
            c["vehicles"] += 1
        elif n in MESSAGE_BOARD:
            c["message_board"] += 1
        elif is_ttc_sign(n):
            c["ttc_signs"] += 1
        elif n in OTHER_ROADWORK:
            c["other_roadwork"] += 1
        else:
            c["other"] += 1
    return dict(c)


def yolo_frame_score(
    names: List[str], weights: Dict[str, float]
) -> Tuple[float, Dict[str, float]]:
    """
    Compute YOLO semantic score from detections.

    Uses weighted combination of semantic group fractions.
    """
    total = len(names)
    gc = group_counts_from_names(names)

    frac_channel = safe_div(gc.get("channelization", 0), total)
    frac_workers = safe_div(gc.get("workers", 0), total)
    frac_vehicles = safe_div(gc.get("vehicles", 0), total)
    frac_ttc = safe_div(gc.get("ttc_signs", 0), total)
    frac_msg = safe_div(gc.get("message_board", 0), total)

    # Weighted linear combination
    raw = 0.0
    raw += weights.get("channelization", 0.0) * frac_channel
    raw += weights.get("workers", 0.0) * frac_workers
    raw += weights.get("vehicles", 0.0) * frac_vehicles
    raw += weights.get("ttc_signs", 0.0) * frac_ttc
    raw += weights.get("message_board", 0.0) * frac_msg
    raw += weights.get("bias", -0.35)

    # Apply logistic transformation
    score = logistic(raw * 4.0)

    feats = {
        "total_objs": float(total),
        "frac_channelization": float(frac_channel),
        "frac_workers": float(frac_workers),
        "frac_vehicles": float(frac_vehicles),
        "frac_ttc_signs": float(frac_ttc),
        "frac_message_board": float(frac_msg),
        "count_channelization": float(gc.get("channelization", 0)),
        "count_workers": float(gc.get("workers", 0)),
        "count_vehicles": float(gc.get("vehicles", 0)),
        "count_ttc_signs": float(gc.get("ttc_signs", 0)),
        "count_message_board": float(gc.get("message_board", 0)),
    }
    return float(score), feats


def update_state(
    prev_state: str,
    fused_score: float,
    inside_frames: int,
    out_frames: int,
    enter_th: float,
    exit_th: float,
    min_inside_frames: int,
    min_out_frames: int,
    approach_th: float,
) -> Tuple[str, int, int]:
    """
    Update state machine for anti-flicker work zone detection.

    States: OUT → APPROACHING → INSIDE → EXITING → OUT
    """
    state = prev_state

    if state == "INSIDE":
        inside_frames += 1
        if fused_score < exit_th and inside_frames >= min_inside_frames:
            state = "EXITING"
            out_frames = 0
        return state, inside_frames, out_frames

    # OUT / APPROACHING / EXITING
    out_frames += 1

    if fused_score >= enter_th and out_frames >= min_out_frames:
        state = "INSIDE"
        inside_frames = 0
        out_frames = 0
        return state, inside_frames, out_frames

    # Otherwise: OUT or APPROACHING
    if fused_score >= approach_th:
        state = "APPROACHING"
    else:
        state = "OUT"

    return state, inside_frames, out_frames


def state_to_label(state: str) -> str:
    """Convert state to display label."""
    if state == "INSIDE":
        return "WORK ZONE"
    if state == "APPROACHING":
        return "APPROACHING"
    if state == "EXITING":
        return "EXITING"
    return "OUTSIDE"


def state_to_color(state: str) -> Tuple[int, int, int]:
    """Convert state to BGR color for OpenCV."""
    if state == "INSIDE":
        return (0, 0, 255)  # Red
    if state == "APPROACHING":
        return (0, 165, 255)  # Orange
    if state == "EXITING":
        return (255, 0, 255)  # Magenta
    return (0, 128, 0)  # Green


def draw_banner(
    frame: np.ndarray, state: str, score: float, clip_active: bool = False
) -> np.ndarray:
    """Draw colored state banner with CLIP indicator."""
    h, w = frame.shape[:2]
    banner_h = int(0.12 * h)
    color = state_to_color(state)
    label = f"{state_to_label(state)}  score={score:.2f}"

    cv2.rectangle(frame, (0, 0), (w, banner_h), color, thickness=-1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    ts, _ = cv2.getTextSize(label, font, scale, thickness)
    x = max(10, (w - ts[0]) // 2)
    y = int(banner_h * 0.72)

    cv2.putText(
        frame, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )

    # CLIP indicator
    if clip_active:
        clip_label = "CLIP ACTIVE"
        c_scale = 0.5
        c_ts, _ = cv2.getTextSize(clip_label, font, c_scale, 1)
        c_x = w - c_ts[0] - 20
        c_y = int(banner_h * 0.5)
        cv2.putText(
            frame, clip_label, (c_x, c_y), font, c_scale, (0, 255, 255), 1, cv2.LINE_AA
        )

    return frame


def list_videos(folder: Path, suffixes=(".mp4", ".mov", ".avi", ".mkv")) -> List[Path]:
    """List video files in folder."""
    if not folder.exists():
        return []
    out = []
    for s in suffixes:
        out += list(folder.glob(f"*{s}"))
    return sorted(out)


def plot_scores(t: List[float], y1: List[float], y2: List[float], title: str) -> None:
    """Plot YOLO and fused scores over time."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, y1, label="yolo_score_ema", linewidth=1.5)
    ax.plot(t, y2, label="fused_score_ema", linewidth=2.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def run_live_preview(
    input_path: Path,
    yolo_model: YOLO,
    device: str,
    conf: float,
    iou: float,
    stride: int,
    ema_alpha: float,
    use_clip: bool,
    clip_bundle: Optional[Dict],
    clip_pos_text: str,
    clip_neg_text: str,
    clip_weight: float,
    clip_trigger_th: float,
    weights_yolo: Dict[str, float],
    enter_th: float,
    exit_th: float,
    min_inside_frames: int,
    min_out_frames: int,
    approach_th: float,
    max_seconds: int,
    run_full_video: bool = False,
    **kwargs
) -> None:
    """Run live preview with real-time rendering."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error(f"Could not open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Timing: stride affects playback speed
    base_dt = 1.0 / fps if fps > 0 else 0.03
    target_dt = base_dt * stride

    # CLIP embeddings
    pos_emb = neg_emb = None
    clip_enabled = False
    if use_clip and clip_bundle is not None:
        try:
            pos_emb, neg_emb = clip_text_embeddings(clip_bundle, device, clip_pos_text, clip_neg_text)
            clip_enabled = True
        except Exception as e:
            logger.warning(f"CLIP text embedding failed: {e}")
            clip_enabled = False

    yolo_ema = None
    fused_ema = None

    state = "OUT"
    inside_frames = 0
    out_frames = 999999

    frame_placeholder = st.empty()
    info_placeholder = st.empty()
    progress = st.progress(0)

    # Stop control
    if "stop_live" not in st.session_state:
        st.session_state["stop_live"] = False

    c_stop1, c_stop2 = st.columns([1, 3])
    with c_stop1:
        if st.button("Stop live preview", type="secondary", key="stop_live_btn"):
            st.session_state["stop_live"] = True
    with c_stop2:
        if st.button("Reset stop flag", key="reset_live_btn"):
            st.session_state["stop_live"] = False

    t_start_loop = time.time()
    frame_idx = 0
    processed = 0

    while True:
        loop_start_time = time.time()

        if (not run_full_video) and max_seconds > 0:
            if (time.time() - t_start_loop) > float(max_seconds):
                break

        ret, frame = cap.read()
        if not ret:
            break

        if st.session_state.get("stop_live", False):
            break

        # Stride skipping
        if stride > 1 and (frame_idx % stride != 0):
            frame_idx += 1
            continue

        # YOLO inference
        results = yolo_model.predict(
            frame,
            conf=conf,
            iou=iou,
            verbose=False,
            device=device,
            half=True,
        )
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.int().cpu().tolist()
            names = [yolo_model.names[int(cid)] for cid in cls_ids]
        else:
            names = []

        yolo_score, feats = yolo_frame_score(names, weights_yolo)

        # Evidence combines object presence and score strength
        total_objs = feats.get("total_objs", 0.0)
        obj_evidence = clamp01(total_objs / 8.0)
        score_evidence = clamp01(yolo_score)
        evidence = clamp01(0.5 * obj_evidence + 0.5 * score_evidence)

        # Adaptive EMA for YOLO score
        alpha_eff = adaptive_alpha(evidence, alpha_min=ema_alpha * 0.4, alpha_max=ema_alpha * 1.2)
        yolo_ema = ema(yolo_ema, yolo_score, alpha_eff)

        # CLIP logic (triggered)
        clip_score_raw = 0.0
        clip_used = 0
        if clip_enabled and (yolo_ema is not None) and (yolo_ema >= clip_trigger_th):
            try:
                diff = clip_frame_score(clip_bundle, device, frame, pos_emb, neg_emb)
                clip_score_raw = logistic(diff * 3.0)
                clip_used = 1
            except Exception as e:
                logger.warning(f"CLIP frame scoring failed: {e}")
                clip_score_raw = 0.0

        # Start with YOLO-only base
        fused = yolo_score
        if clip_enabled and (yolo_ema is not None) and (yolo_ema >= clip_trigger_th):
            fused = (1.0 - clip_weight) * fused + clip_weight * clip_score_raw

        # Lightweight context cue: orange pixels boost when uncertain
        if kwargs.get("enable_context_boost", False) and (yolo_ema is not None) and (yolo_ema < kwargs.get("context_trigger_below", 0.55)):
            ratio = orange_ratio_hsv(
                frame,
                h_low=int(kwargs.get("orange_h_low", 5)),
                h_high=int(kwargs.get("orange_h_high", 25)),
                s_th=int(kwargs.get("orange_s_th", 80)),
                v_th=int(kwargs.get("orange_v_th", 50)),
            )
            ctx_score = context_boost_from_orange(ratio, center=float(kwargs.get("orange_center", 0.08)), k=float(kwargs.get("orange_k", 30.0)))
            cw = float(kwargs.get("orange_weight", 0.25))
            fused = (1.0 - cw) * fused + cw * ctx_score

        fused = clamp01(fused)

        # Adaptive EMA for fused score
        alpha_eff_fused = adaptive_alpha(evidence, alpha_min=ema_alpha * 0.4, alpha_max=ema_alpha * 1.2)
        fused_ema = ema(fused_ema, fused, alpha_eff_fused)

        state, inside_frames, out_frames = update_state(
            prev_state=state,
            fused_score=float(fused_ema),
            inside_frames=inside_frames,
            out_frames=out_frames,
            enter_th=enter_th,
            exit_th=exit_th,
            min_inside_frames=min_inside_frames,
            min_out_frames=min_out_frames,
            approach_th=approach_th,
        )

        annotated = r.plot()
        annotated = draw_banner(annotated, state, float(fused_ema), clip_active=(clip_used == 1))

        # Resize for browser performance
        disp_h, disp_w = annotated.shape[:2]
        if disp_w > 1280:
            scale = 1280 / disp_w
            annotated = cv2.resize(annotated, (1280, int(disp_h * scale)))

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB", width='stretch')

        t_sec = float(frame_idx / fps) if fps > 0 else float(processed)
        info_placeholder.markdown(
            f"**Frame:** {frame_idx} / {total_frames} | "
            f"**t:** {t_sec:.2f}s | **state:** `{state}` | "
            f"**fused_ema:** {float(fused_ema) if fused_ema else 0.0:.2f} | "
            f"**CLIP:** {'ACTIVE' if clip_used else 'OFF'}"
        )

        if total_frames > 0:
            progress.progress(min(frame_idx / total_frames, 1.0))

        processed += 1
        frame_idx += 1

        # Smart sleep based on stride
        process_duration = time.time() - loop_start_time
        sleep_time = target_dt - process_duration
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    st.success(f"Live preview finished. Processed steps: {processed}.")


def process_video(
    input_path: Path,
    yolo_model: YOLO,
    device: str,
    conf: float,
    iou: float,
    stride: int,
    ema_alpha: float,
    use_clip: bool,
    clip_bundle: Optional[Dict],
    clip_pos_text: str,
    clip_neg_text: str,
    clip_weight: float,
    clip_trigger_th: float,
    weights_yolo: Dict[str, float],
    enter_th: float,
    exit_th: float,
    min_inside_frames: int,
    min_out_frames: int,
    approach_th: float,
    save_video: bool,
) -> Dict:
    """
    Process video with YOLO + CLIP fusion and state machine.

    Returns dict with output paths, timeline DataFrame, and metrics.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Output paths
    tmp_dir = Path(tempfile.gettempdir())
    out_video_path = tmp_dir / f"{input_path.stem}_annotated_fusion.mp4"
    out_csv_path = tmp_dir / f"{input_path.stem}_timeline_fusion.csv"

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        effective_fps = fps / stride
        writer = cv2.VideoWriter(
            str(out_video_path), fourcc, float(effective_fps), (width, height)
        )

    # CLIP embeddings
    pos_emb = neg_emb = None
    clip_enabled = False
    if use_clip and clip_bundle is not None:
        try:
            pos_emb, neg_emb = clip_text_embeddings(
                clip_bundle, device, clip_pos_text, clip_neg_text
            )
            clip_enabled = True
        except Exception as e:
            logger.error(f"CLIP embedding error: {e}")

    timeline_rows = []
    yolo_ema = None
    fused_ema = None

    state = "OUT"
    inside_frames = 0
    out_frames = 999999

    progress = st.progress(0)
    info = st.empty()

    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Stride skipping
        if stride > 1 and (frame_idx % stride != 0):
            frame_idx += 1
            continue

        # YOLO inference
        results = yolo_model.predict(
            frame, conf=conf, iou=iou, verbose=False, device=device, half=True
        )
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.int().cpu().tolist()
            names = [yolo_model.names[int(cid)] for cid in cls_ids]
        else:
            names = []

        # YOLO score
        yolo_score, feats = yolo_frame_score(names, weights_yolo)

        # Evidence combines object presence and score strength
        total_objs = feats.get("total_objs", 0.0)
        obj_evidence = clamp01(total_objs / 8.0)
        score_evidence = clamp01(yolo_score)
        evidence = clamp01(0.5 * obj_evidence + 0.5 * score_evidence)

        # Adaptive EMA for YOLO score
        alpha_eff = adaptive_alpha(evidence, alpha_min=ema_alpha * 0.4, alpha_max=ema_alpha * 1.2)
        yolo_ema = ema(yolo_ema, yolo_score, alpha_eff)

        # CLIP (triggered)
        clip_score_raw = 0.0
        clip_used = 0
        if clip_enabled and (yolo_ema is not None) and (yolo_ema >= clip_trigger_th):
            try:
                diff = clip_frame_score(clip_bundle, device, frame, pos_emb, neg_emb)
                clip_score_raw = logistic(diff * 3.0)
                clip_used = 1
            except Exception as e:
                logger.error(f"CLIP frame score error: {e}")

        # Fuse scores starting from YOLO
        fused = yolo_score
        if clip_enabled and (yolo_ema is not None) and (yolo_ema >= clip_trigger_th):
            fused = (1.0 - clip_weight) * fused + clip_weight * clip_score_raw

        # Lightweight context cue: orange pixels boost when uncertain
        if st.session_state.get("enable_context_boost", False) and (yolo_ema is not None) and (yolo_ema < st.session_state.get("context_trigger_below", 0.55)):
            ratio = orange_ratio_hsv(
                frame,
                h_low=int(st.session_state.get("orange_h_low", 5)),
                h_high=int(st.session_state.get("orange_h_high", 25)),
                s_th=int(st.session_state.get("orange_s_th", 80)),
                v_th=int(st.session_state.get("orange_v_th", 50)),
            )
            ctx_score = context_boost_from_orange(
                ratio,
                center=float(st.session_state.get("orange_center", 0.08)),
                k=float(st.session_state.get("orange_k", 30.0)),
            )
            cw = float(st.session_state.get("orange_weight", 0.25))
            fused = (1.0 - cw) * fused + cw * ctx_score

        fused = clamp01(fused)
        # Adaptive EMA for fused score
        alpha_eff_fused = adaptive_alpha(evidence, alpha_min=ema_alpha * 0.4, alpha_max=ema_alpha * 1.2)
        fused_ema = ema(fused_ema, fused, alpha_eff_fused)

        # State update
        state, inside_frames, out_frames = update_state(
            prev_state=state,
            fused_score=float(fused_ema),
            inside_frames=inside_frames,
            out_frames=out_frames,
            enter_th=enter_th,
            exit_th=exit_th,
            min_inside_frames=min_inside_frames,
            min_out_frames=min_out_frames,
            approach_th=approach_th,
        )

        # Annotate
        annotated = r.plot()
        annotated = draw_banner(
            annotated, state, float(fused_ema), clip_active=(clip_used == 1)
        )

        if writer is not None:
            writer.write(annotated)

        # Save timeline row
        time_sec = float(frame_idx / fps) if fps > 0 else float(processed)
        row = {
            "frame": int(frame_idx),
            "time_sec": float(time_sec),
            "yolo_score": float(yolo_score),
            "yolo_score_ema": (
                float(yolo_ema) if yolo_ema is not None else float(yolo_score)
            ),
            "fused_score_ema": (
                float(fused_ema) if fused_ema is not None else float(fused)
            ),
            "state": str(state),
            "clip_used": int(clip_used),
            "clip_score": float(clip_score_raw),
            "count_channelization": int(feats["count_channelization"]),
            "count_workers": int(feats["count_workers"]),
        }
        timeline_rows.append(row)

        processed += 1
        frame_idx += 1

        # UI updates
        if processed % 10 == 0:
            if total_frames > 0:
                progress.progress(min(frame_idx / total_frames, 1.0))
            info.markdown(
                f"**Processing...** Frame {frame_idx}/{total_frames} | "
                f"State: `{state}` | CLIP used: {clip_used}"
            )

    cap.release()
    if writer is not None:
        writer.release()
    progress.progress(1.0)

    df = pd.DataFrame(timeline_rows)
    df.to_csv(out_csv_path, index=False)

    return {
        "out_video_path": out_video_path if save_video else None,
        "out_csv_path": out_csv_path,
        "timeline_df": df,
        "fps": fps,
        "processed": processed,
        "total_frames": total_frames,
    }


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(page_title="Work Zone Semantic Fusion", layout="wide")
    st.title("Work Zone Detection - Semantic Fusion (YOLO + CLIP)")

    st.markdown(
        """
        This app combines **YOLO semantic scoring** with **CLIP verification** and an **anti-flicker state machine**:
        - YOLO: Detects work zone objects and computes semantic score
        - CLIP: Triggered semantic verification when YOLO confidence is high
        - State machine: OUT → APPROACHING → INSIDE → EXITING (prevents flickering)
        """
    )

    # Sidebar
    st.sidebar.header("Model + Runtime")

    device_choice = st.sidebar.radio(
        "Device", ["Auto", "GPU (cuda)", "CPU"], index=0
    )
    device = resolve_device(device_choice)
    st.sidebar.write(f"Using: **{device}**")

    st.sidebar.markdown("---")
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "YOLO Model",
        [
            "Hard-Negative Trained (latest)",
            "Fusion Baseline (pre hard-neg)",
            "Upload Custom Weights"
        ],
        index=0
    )
    
    if model_choice == "Hard-Negative Trained (latest)":
        selected_weights = HARDNEG_WEIGHTS
        st.sidebar.info("✅ Using model trained with 134 hard negatives (84.6% false positive reduction)")
        use_uploaded_weights = False
    elif model_choice == "Fusion Baseline (pre hard-neg)":
        selected_weights = FUSION_BASELINE_WEIGHTS
        st.sidebar.info("Using baseline fusion model")
        use_uploaded_weights = False
    else:
        selected_weights = None
        use_uploaded_weights = True

    st.sidebar.markdown("---")
    mode = st.sidebar.radio("Run mode", ["Live preview (real time)", "Batch (save outputs)"], index=0)
    max_seconds = st.sidebar.number_input("Live preview max seconds", min_value=5, value=30, step=5)
    run_full_video = st.sidebar.checkbox("Run full video (ignore max seconds)", value=True)

    if use_uploaded_weights:
        uploaded_weights = st.sidebar.file_uploader("Upload weights (.pt)", type=["pt"])
    else:
        if selected_weights:
            st.sidebar.text("Selected weights:")
            st.sidebar.code(selected_weights)
        uploaded_weights = None

    conf = st.sidebar.slider("Confidence", 0.05, 0.90, 0.25, 0.05)
    iou = st.sidebar.slider("IoU", 0.10, 0.90, 0.70, 0.05)
    stride = st.sidebar.number_input("Frame stride", 1, 30, 2, 1)
    ema_alpha = st.sidebar.slider("EMA alpha", 0.05, 0.60, 0.25, 0.01)

    st.sidebar.markdown("---")
    st.sidebar.header("YOLO semantic weights")
    w_bias = st.sidebar.slider("bias", -1.0, 0.5, -0.35, 0.05)
    w_channel = st.sidebar.slider("channelization", 0.0, 2.0, 0.9, 0.05)
    w_workers = st.sidebar.slider("workers", 0.0, 2.0, 0.8, 0.05)
    w_vehicles = st.sidebar.slider("vehicles", 0.0, 2.0, 0.5, 0.05)
    w_ttc = st.sidebar.slider("ttc_signs", 0.0, 2.0, 0.7, 0.05)
    w_msg = st.sidebar.slider("message_board", 0.0, 2.0, 0.6, 0.05)

    weights_yolo = {
        "bias": float(w_bias),
        "channelization": float(w_channel),
        "workers": float(w_workers),
        "vehicles": float(w_vehicles),
        "ttc_signs": float(w_ttc),
        "message_board": float(w_msg),
    }

    st.sidebar.markdown("---")
    st.sidebar.header("State machine")
    enter_th = st.sidebar.slider("Enter threshold", 0.50, 0.95, 0.70, 0.01)
    exit_th = st.sidebar.slider("Exit threshold", 0.05, 0.70, 0.45, 0.01)
    approach_th = st.sidebar.slider("Approach threshold", 0.10, 0.90, 0.55, 0.01)
    min_inside_frames = st.sidebar.number_input("Min INSIDE frames", 1, value=25, step=5)
    min_out_frames = st.sidebar.number_input("Min OUT frames", 1, value=15, step=5)

    st.sidebar.markdown("---")
    st.sidebar.header("CLIP fusion")
    use_clip = st.sidebar.checkbox("Enable CLIP", value=True)

    clip_pos_text = st.sidebar.text_input(
        "Positive prompt",
        value="a road work zone with traffic cones, barriers, workers, construction signs",
    )
    clip_neg_text = st.sidebar.text_input(
        "Negative prompt", value="a normal road with no construction and no work zone"
    )
    clip_weight = st.sidebar.slider("CLIP weight", 0.0, 0.8, 0.35, 0.05)
    clip_trigger_th = st.sidebar.slider("CLIP trigger (YOLO≥)", 0.0, 1.0, 0.45, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.header("Fusion boosts")
    enable_context_boost = st.sidebar.checkbox("Enable orange-cue boost on low evidence", value=True)
    orange_weight = st.sidebar.slider("Orange boost weight", 0.0, 0.6, 0.25, 0.05)
    context_trigger_below = st.sidebar.slider("Apply boost if YOLO_ema <", 0.0, 1.0, 0.55, 0.05)

    with st.sidebar.expander("Advanced (HSV and shaping)"):
        orange_h_low = st.slider("Hue low", 0, 179, 5, 1)
        orange_h_high = st.slider("Hue high", 0, 179, 25, 1)
        orange_s_th = st.slider("Saturation min", 0, 255, 80, 5)
        orange_v_th = st.slider("Value min", 0, 255, 50, 5)
        orange_center = st.slider("Center ratio (0-0.3)", 0.00, 0.30, 0.08, 0.01)
        orange_k = st.slider("Slope (k)", 1.0, 60.0, 30.0, 1.0)

    # Store context boost params in session_state for batch mode
    st.session_state["enable_context_boost"] = enable_context_boost
    st.session_state["orange_weight"] = orange_weight
    st.session_state["context_trigger_below"] = context_trigger_below
    st.session_state["orange_h_low"] = orange_h_low
    st.session_state["orange_h_high"] = orange_h_high
    st.session_state["orange_s_th"] = orange_s_th
    st.session_state["orange_v_th"] = orange_v_th
    st.session_state["orange_center"] = orange_center
    st.session_state["orange_k"] = orange_k

    save_video = st.sidebar.checkbox("Save annotated video", value=True)

    # Main area
    st.subheader("Video Source")
    source_choice = st.radio(
        "Source", ["Demo videos", "Dataset videos", "Upload"], index=0
    )

    video_path = None

    if source_choice == "Upload":
        vid = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
        if vid is not None:
            tmp = Path(tempfile.gettempdir()) / vid.name
            with open(tmp, "wb") as f:
                f.write(vid.getbuffer())
            video_path = tmp
    else:
        folder = (
            DEMO_VIDEOS_DIR if "Demo" in source_choice else VIDEOS_COMPRESSED_DIR
        )
        vids = list_videos(folder)
        if not vids:
            st.warning(f"No videos found in: {folder}")
        else:
            names = [p.name for p in vids]
            chosen = st.selectbox("Choose video", names)
            video_path = folder / chosen

    run_btn = st.button("Run Detection", type="primary")

    if not run_btn:
        return

    if video_path is None:
        st.error("Please select/upload a video first.")
        return

    # Load YOLO
    with st.spinner("Loading YOLO model..."):
        try:
            if use_uploaded_weights:
                if uploaded_weights is None:
                    st.error("Please upload weights")
                    return
                yolo_model = load_model_cached(
                    uploaded_weights.read(),
                    Path(uploaded_weights.name).suffix,
                    device,
                )
            else:
                if not Path(selected_weights).exists():
                    st.error(f"Weights not found: {selected_weights}")
                    return
                yolo_model = load_model_default(selected_weights, device)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

    st.success(f"YOLO loaded on {device}")

    # Load CLIP
    clip_bundle = None
    clip_ok = False
    if use_clip:
        with st.spinner("Loading CLIP..."):
            clip_ok, clip_bundle = load_clip_bundle(device=device)
        if clip_ok:
            st.success("CLIP enabled (OpenCLIP)")
        else:
            st.warning("CLIP not available. Continuing without CLIP.")
            use_clip = False

    # Process
    if mode == "Live preview (real time)":
        st.info("Running live preview (real time).")
        run_live_preview(
            input_path=video_path,
            yolo_model=yolo_model,
            device=device,
            conf=float(conf),
            iou=float(iou),
            stride=int(stride),
            ema_alpha=float(ema_alpha),
            use_clip=bool(use_clip),
            clip_bundle=clip_bundle,
            clip_pos_text=str(clip_pos_text),
            clip_neg_text=str(clip_neg_text),
            clip_weight=float(clip_weight),
            clip_trigger_th=float(clip_trigger_th),
            weights_yolo=weights_yolo,
            enter_th=float(enter_th),
            exit_th=float(exit_th),
            min_inside_frames=int(min_inside_frames),
            min_out_frames=int(min_out_frames),
            approach_th=float(approach_th),
            max_seconds=int(max_seconds),
            run_full_video=bool(run_full_video),
            enable_context_boost=bool(enable_context_boost),
            orange_weight=float(orange_weight),
            context_trigger_below=float(context_trigger_below),
            orange_h_low=int(orange_h_low),
            orange_h_high=int(orange_h_high),
            orange_s_th=int(orange_s_th),
            orange_v_th=int(orange_v_th),
            orange_center=float(orange_center),
            orange_k=float(orange_k),
        )
        return

    # Batch processing
    with st.spinner("Processing video..."):
        result = process_video(
            input_path=video_path,
            yolo_model=yolo_model,
            device=device,
            conf=float(conf),
            iou=float(iou),
            stride=int(stride),
            ema_alpha=float(ema_alpha),
            use_clip=bool(use_clip),
            clip_bundle=clip_bundle,
            clip_pos_text=str(clip_pos_text),
            clip_neg_text=str(clip_neg_text),
            clip_weight=float(clip_weight),
            clip_trigger_th=float(clip_trigger_th),
            weights_yolo=weights_yolo,
            enter_th=float(enter_th),
            exit_th=float(exit_th),
            min_inside_frames=int(min_inside_frames),
            min_out_frames=int(min_out_frames),
            approach_th=float(approach_th),
            save_video=bool(save_video),
        )

    df = result["timeline_df"]
    st.success("Processing complete.")

    # Results
    st.subheader("Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processed steps", int(result["processed"]))
    c2.metric("FPS", f"{result['fps']:.1f}")
    c3.metric("Stride", int(stride))
    c4.metric(
        "CLIP used (frames)",
        int(df["clip_used"].sum()) if "clip_used" in df.columns else 0,
    )

    # Score plot
    st.subheader("Score curves")
    t = df["time_sec"].tolist()
    y_yolo = df["yolo_score_ema"].tolist()
    y_fused = df["fused_score_ema"].tolist()
    plot_scores(t, y_yolo, y_fused, title=f"Scores - {video_path.name}")

    # State distribution
    st.subheader("State counts")
    st.write(df["state"].value_counts())

    # CSV download
    st.subheader("Timeline CSV")
    st.dataframe(df.head(30))

    with open(result["out_csv_path"], "rb") as f:
        st.download_button(
            "Download timeline CSV",
            data=f,
            file_name=Path(result["out_csv_path"]).name,
            mime="text/csv",
        )

    # Video
    if result["out_video_path"] is not None:
        st.subheader("Annotated video")
        st.video(str(result["out_video_path"]))
        with open(result["out_video_path"], "rb") as f:
            st.download_button(
                "Download annotated video",
                data=f,
                file_name=Path(result["out_video_path"]).name,
                mime="video/mp4",
            )


if __name__ == "__main__":
    main()
